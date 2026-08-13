import concurrent.futures
import copy
import dataclasses
import logging
import threading
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

# Wire names for the real-time chunking control fields. These MUST match the constants in
# `openpi.policies.policy` on the server; `rtc_protocol_test.py` in the server repo asserts it,
# because a rename here would otherwise silently disable guidance rather than raise (the server
# drops observation keys it does not recognize).
RTC_ENABLED = "rtc/enabled"
RTC_PREV_CHUNK = "rtc/prev_chunk"
RTC_INFERENCE_DELAY = "rtc/inference_delay"
RTC_PREFIX_ATTENTION_HORIZON = "rtc/prefix_attention_horizon"
RTC_PREFIX_ATTENTION_SCHEDULE = "rtc/prefix_attention_schedule"
RTC_MAX_GUIDANCE_WEIGHT = "rtc/max_guidance_weight"
RTC_RAW_ACTIONS = "rtc/raw_actions"

logger = logging.getLogger(__name__)


def _snapshot_obs(obs: Dict) -> Dict:  # noqa: UP006
    """Copy an observation before handing it to a background inference call."""

    def copier(x):
        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)
        return copy.deepcopy(x)

    return tree.map_structure(copier, obs)


def _infer_policy(policy: _base_policy.BasePolicy, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
    if noise is None:
        return policy.infer(obs)
    return policy.infer(obs, noise)


def _slice_step(results: Dict, step: int) -> Dict:  # noqa: UP006
    def slicer(x):
        if isinstance(x, np.ndarray):
            return x[step, ...]
        else:
            return x

    return tree.map_structure(slicer, results)


def _chunk_len(results: Dict) -> int:  # noqa: UP006
    actions = results.get("actions")
    if not isinstance(actions, np.ndarray):
        raise ValueError("Policy output must contain an 'actions' numpy array.")
    if actions.ndim == 0:
        raise ValueError("Policy output 'actions' must have a leading chunk dimension.")
    return actions.shape[0]


def _shift_chunk(chunk: np.ndarray, shift: int) -> np.ndarray:
    """Re-base a chunk in time: entry `i` of the result is entry `i + shift` of the input.

    The tail has no data left, so it is zero-filled. That is safe only because the caller also
    caps the prefix attention horizon at `len(chunk) - shift`, which forces the guidance weights
    over the padding to be exactly zero -- otherwise the new chunk would be guided toward zeros.
    """
    if shift <= 0:
        return chunk
    if shift >= chunk.shape[0]:
        return np.zeros_like(chunk)
    return np.concatenate([chunk[shift:], np.zeros_like(chunk[:shift])], axis=0)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = _infer_policy(self._policy, obs, noise)
            self._cur_step = 0

        results = _slice_step(self._last_results, self._cur_step)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0

    @override
    def get_prefix_rep(self, observation: Dict) -> Dict:
        return self._policy.get_prefix_rep(observation)


@dataclasses.dataclass
class _PendingRequest:
    """An in-flight inference, plus the time base the returned chunk is aligned to."""

    future: concurrent.futures.Future
    origin: int  # global control step that the new chunk's index 0 corresponds to


class RealTimeActionChunkBroker(_base_policy.BasePolicy):
    """Dispenses one action per call, replanning continuously with real-time chunking.

    Implements the client half of RTC (Black et al., 2025, "Real-Time Execution of Action Chunking
    Flow Policies"); the server half is the guidance in `Pi0.sample_actions`. The two halves are
    useless apart: guidance without correct client-side time alignment conditions on the wrong
    actions, and alignment without guidance just splices unrelated chunks together.

    The loop, with execute horizon `e`, inference delay `d`, and model action horizon `H`:

      * Every `e` control steps, fire a background inference. The request carries the chunk
        currently being executed, time-shifted so its index 0 lines up with *now*, and asks the
        server to pin the new chunk's first `d` actions to it (those are the actions that will be
        executed from the current chunk while inference runs) and to decay guidance to zero by
        index `H - e`.
      * When the result lands, splice it in at the index matching the elapsed control steps -- NOT
        at index 0. The chunk is a plan on an absolute time base; restarting it at 0 would replay
        actions aimed at a pose the robot already left, which reads as a backward jerk at every
        replan.

    So each chunk is executed over its indices `[d, d + e)`, which requires `d <= e` and
    `e + d <= H`; both are checked once the model's horizon is known.

    Timing is measured, not assumed: `d` is what the server is *told* to pin, while the actual
    splice index comes from the step counter. A chunk that arrives early is installed early (safe:
    its prefix is pinned to what is already executing), and one that arrives late is spliced
    further in, with a warning -- late arrivals land past the pinned prefix, where guidance was
    only soft, so continuity degrades and `inference_delay` should be raised.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        execute_horizon: Optional[int] = None,
        inference_delay: int = 1,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        exhausted_behavior: str = "hold",
    ):
        """
        Args:
            policy: The inner policy, typically a `WebsocketClientPolicy`.
            execute_horizon: Replan period in control steps. Defaults to half the model's action
                horizon once that is known.
            inference_delay: How many control steps inference is expected to take. This is the
                length of the hard-pinned prefix, so it should be an upper bound on the real
                latency (round trip, not just GPU time) at the control rate -- rounding up costs
                a little reactivity, rounding down costs continuity.
            prefix_attention_schedule: Decay shape between `inference_delay` and the prefix
                attention horizon. One of "linear", "exp", "ones", "zeros".
            max_guidance_weight: Clip on the guidance weight, which diverges at both ends of the
                denoising trajectory.
            exhausted_behavior: What to do if a chunk runs out before its replacement arrives:
                "hold" repeats the last action, "block" waits. Either way, if nothing is in
                flight, a synchronous inference is issued rather than stalling forever.
        """
        if execute_horizon is not None and execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive.")
        if inference_delay < 0:
            raise ValueError("inference_delay must be non-negative.")
        if execute_horizon is not None and inference_delay > execute_horizon:
            raise ValueError(
                f"inference_delay ({inference_delay}) must not exceed execute_horizon "
                f"({execute_horizon}); the next chunk has to arrive before it is needed."
            )
        if prefix_attention_schedule not in ("linear", "exp", "ones", "zeros"):
            raise ValueError(f"Invalid prefix_attention_schedule: {prefix_attention_schedule}")
        if exhausted_behavior not in ("hold", "block"):
            raise ValueError("exhausted_behavior must be either 'hold' or 'block'.")

        self._policy = policy
        self._execute_horizon = execute_horizon
        self._inference_delay = inference_delay
        self._prefix_attention_schedule = prefix_attention_schedule
        self._max_guidance_weight = max_guidance_weight
        self._exhausted_behavior = exhausted_behavior

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._policy_lock = threading.Lock()

        self._step: int = 0  # monotonic control-step counter; the shared time base
        self._origin: int = 0  # global step corresponding to index 0 of the current chunk
        self._results: Optional[Dict[str, Any]] = None
        self._raw_chunk: Optional[np.ndarray] = None  # model-space chunk, echoed back for guidance
        self._last_result: Optional[Dict] = None  # noqa: UP006
        self._last_fire_step: int = 0
        self._pending: Optional[_PendingRequest] = None
        self._validated: bool = False

    @override
    def infer(self, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
        self._install_pending(block=False)

        if not self._has_action():
            # The current chunk ran out before its replacement landed. Only hold/block while a
            # request is actually in flight -- otherwise there is nothing to wait for, and holding
            # would freeze the arm on a stale action forever.
            if self._pending is not None:
                if self._exhausted_behavior == "hold" and self._last_result is not None:
                    logger.warning(
                        "RTC: chunk exhausted at step %d with inference still in flight; holding "
                        "the last action. Increase execute_horizon or reduce the control rate.",
                        self._step,
                    )
                    self._step += 1
                    return self._last_result
                self._install_pending(block=True)

        if not self._has_action():
            self._install(self._infer_sync(obs, noise, prev=None), origin=self._step)

        self._maybe_prefetch(obs, noise)

        result = _slice_step(self._results, self._step - self._origin)
        self._last_result = result
        self._step += 1
        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        # Drop any in-flight request: it was computed against the pre-reset observation, and after
        # a handback that plan aims at the pose the operator just corrected away from. Dropping the
        # reference is what discards it -- cancel() is best-effort and loses to an already-running
        # call, whose result simply goes nowhere.
        if self._pending is not None:
            self._pending.future.cancel()
        self._pending = None
        self._results = None
        self._raw_chunk = None
        self._last_result = None
        self._step = 0
        self._origin = 0
        self._last_fire_step = 0

    @override
    def get_prefix_rep(self, observation: Dict) -> Dict:
        return self._policy.get_prefix_rep(observation)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    # -- internals ------------------------------------------------------------

    def _has_action(self) -> bool:
        return self._results is not None and 0 <= self._step - self._origin < _chunk_len(self._results)

    def _infer_sync(self, obs: Dict, noise: float, prev: Optional[Dict]) -> Dict:  # noqa: UP006
        request = self._build_request(obs, prev)
        # Most robot-side policy clients own a single websocket connection, so foreground and
        # background calls must be serialized even when a reset races an in-flight request.
        with self._policy_lock:
            return _infer_policy(self._policy, request, noise)

    def _build_request(self, obs: Dict, prev: Optional[Dict]) -> Dict:  # noqa: UP006
        request = dict(obs)
        request[RTC_ENABLED] = True
        if prev is not None:
            request[RTC_PREV_CHUNK] = prev["chunk"]
            request[RTC_INFERENCE_DELAY] = prev["delay"]
            request[RTC_PREFIX_ATTENTION_HORIZON] = prev["horizon"]
            request[RTC_PREFIX_ATTENTION_SCHEDULE] = self._prefix_attention_schedule
            request[RTC_MAX_GUIDANCE_WEIGHT] = self._max_guidance_weight
        return request

    def _maybe_prefetch(self, obs: Dict, noise: float) -> None:  # noqa: UP006
        if self._pending is not None:
            return
        if self._step - self._last_fire_step < self._effective_execute_horizon():
            return

        prev = None
        if self._raw_chunk is not None:
            shift = self._step - self._origin
            chunk_len = self._raw_chunk.shape[0]
            prev = {
                "chunk": _shift_chunk(self._raw_chunk, shift),
                "delay": self._inference_delay,
                # Past `chunk_len - shift` the shifted chunk is zero padding, so guidance must
                # already have decayed to zero by then.
                "horizon": max(0, chunk_len - shift),
            }

        origin = self._step
        self._last_fire_step = origin
        future = self._executor.submit(self._infer_sync, _snapshot_obs(obs), noise, prev)
        self._pending = _PendingRequest(future=future, origin=origin)

    def _install_pending(self, *, block: bool) -> bool:
        pending = self._pending
        if pending is None:
            return False
        if not block and not pending.future.done():
            return False

        self._pending = None
        try:
            results = pending.future.result()
        except Exception:
            # Never let a dropped websocket propagate into the control loop with the arm driven;
            # the caller falls back to a synchronous inference, which will raise more visibly if
            # the connection is really gone.
            logger.exception("RTC: background inference failed; falling back to synchronous inference.")
            return False

        index = self._step - pending.origin
        chunk_len = _chunk_len(results)
        if index >= chunk_len:
            logger.warning(
                "RTC: chunk arrived %d steps late and is fully expired (horizon %d); discarding.",
                index,
                chunk_len,
            )
            return False
        if index > self._inference_delay:
            logger.warning(
                "RTC: chunk arrived at index %d but only the first %d actions were pinned. "
                "Continuity is not guaranteed past the pinned prefix -- raise inference_delay.",
                index,
                self._inference_delay,
            )

        self._install(results, origin=pending.origin)
        return True

    def _install(self, results: Dict, *, origin: int) -> None:  # noqa: UP006
        raw = results.pop(RTC_RAW_ACTIONS, None)
        if raw is None:
            raise ValueError(
                f"Policy response is missing '{RTC_RAW_ACTIONS}'. The server does not support "
                "real-time chunking (it is echoed whenever the request sets "
                f"'{RTC_ENABLED}'); update the openpi server, or use ActionChunkBroker instead."
            )
        self._raw_chunk = np.asarray(raw)
        self._results = results
        self._origin = origin
        self._validate_once(_chunk_len(results))

    def _effective_execute_horizon(self) -> int:
        if self._execute_horizon is not None:
            return self._execute_horizon
        if self._results is None:
            return 1
        return max(1, _chunk_len(self._results) // 2)

    def _validate_once(self, action_horizon: int) -> None:
        """Check the timing budget against the model's real action horizon.

        Deferred to the first response because the horizon is a property of the served checkpoint,
        not of this broker. Getting it wrong is not a crash, it is a chunk that expires mid-flight,
        so it is worth failing loudly at the start of a rollout rather than mid-episode.
        """
        if self._validated:
            return
        self._validated = True
        execute_horizon = self._effective_execute_horizon()
        if self._inference_delay > execute_horizon:
            raise ValueError(
                f"inference_delay ({self._inference_delay}) must not exceed execute_horizon "
                f"({execute_horizon})."
            )
        if execute_horizon + self._inference_delay > action_horizon:
            raise ValueError(
                f"execute_horizon ({execute_horizon}) + inference_delay ({self._inference_delay}) "
                f"exceeds the model's action horizon ({action_horizon}): each chunk is executed "
                f"over indices [{self._inference_delay}, {execute_horizon + self._inference_delay}) "
                "and would run off the end. Lower execute_horizon."
            )
