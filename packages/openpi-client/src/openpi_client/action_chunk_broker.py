import concurrent.futures
import copy
import logging
import threading
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


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


def _validate_horizon(results: Dict, action_horizon: int) -> None:  # noqa: UP006
    chunk_len = _chunk_len(results)
    if chunk_len < action_horizon:
        raise ValueError(f"Policy returned {chunk_len} actions, but broker action_horizon is {action_horizon}.")


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


class RealTimeActionChunkBroker(_base_policy.BasePolicy):
    """Returns one action per call while prefetching the next action chunk.

    The first call blocks because no action is available yet. After that, a
    background inference call starts when the current chunk has ``replan_margin``
    actions left. If the current chunk is exhausted before the next one is ready,
    the broker either holds the last action or blocks, depending on
    ``exhausted_behavior``.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: Optional[int] = None,
        replan_margin: Optional[int] = None,
        exhausted_behavior: str = "hold",
    ):
        if action_horizon is not None and action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if replan_margin is not None and replan_margin < 0:
            raise ValueError("replan_margin must be non-negative.")
        if exhausted_behavior not in ("hold", "block"):
            raise ValueError("exhausted_behavior must be either 'hold' or 'block'.")

        self._policy = policy
        self._action_horizon = action_horizon
        self._replan_margin = replan_margin
        self._exhausted_behavior = exhausted_behavior

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._policy_lock = threading.Lock()
        self._cur_step: int = 0
        self._generation: int = 0
        self._last_result: Optional[Dict] = None  # noqa: UP006
        self._current_results: Optional[Dict[str, Any]] = None
        self._next_future: Optional[concurrent.futures.Future] = None

    @override
    def infer(self, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
        if self._current_results is None:
            installed = self._install_pending_if_ready(block=self._exhausted_behavior == "block")
            if not installed:
                if self._last_result is not None and self._exhausted_behavior == "hold":
                    return self._last_result
                self._install_chunk(self._infer(_snapshot_obs(obs), noise))

        action_horizon = self._effective_horizon()
        remaining = action_horizon - self._cur_step
        if self._next_future is None and remaining <= self._effective_replan_margin(action_horizon):
            self._next_future = self._submit_inference(obs, noise)

        result = _slice_step(self._current_results, self._cur_step)
        self._last_result = result
        self._cur_step += 1

        if self._cur_step >= action_horizon and not self._install_pending_if_ready(block=False):
            self._current_results = None

        return result

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._generation += 1
        if self._next_future is not None:
            self._next_future.cancel()
        self._next_future = None
        self._current_results = None
        self._last_result = None
        self._cur_step = 0

    @override
    def get_prefix_rep(self, observation: Dict) -> Dict:
        return self._policy.get_prefix_rep(observation)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _infer(self, obs: Dict, noise: float = None) -> Dict:  # noqa: UP006
        # Most robot-side policy clients own a single websocket connection. Keep
        # foreground and background calls serialized even when reset races a request.
        with self._policy_lock:
            return _infer_policy(self._policy, obs, noise)

    def _submit_inference(self, obs: Dict, noise: float = None) -> concurrent.futures.Future:  # noqa: UP006
        generation = self._generation
        obs_snapshot = _snapshot_obs(obs)
        return self._executor.submit(self._infer_with_generation, generation, obs_snapshot, noise)

    def _infer_with_generation(self, generation: int, obs: Dict, noise: float = None) -> Tuple[int, Dict]:  # noqa: UP006
        return generation, self._infer(obs, noise)

    def _install_pending_if_ready(self, *, block: bool) -> bool:
        if self._next_future is None:
            return False
        if not block and not self._next_future.done():
            return False

        future = self._next_future
        self._next_future = None
        generation, results = future.result()
        if generation != self._generation:
            logging.debug("Dropping stale action chunk from before reset.")
            return False
        self._install_chunk(results)
        return True

    def _install_chunk(self, results: Dict) -> None:  # noqa: UP006
        if self._action_horizon is not None:
            _validate_horizon(results, self._action_horizon)
        self._current_results = results
        self._cur_step = 0

    def _effective_horizon(self) -> int:
        assert self._current_results is not None
        if self._action_horizon is not None:
            return self._action_horizon
        return _chunk_len(self._current_results)

    def _effective_replan_margin(self, action_horizon: int) -> int:
        replan_margin = self._replan_margin if self._replan_margin is not None else max(1, action_horizon // 2)
        if replan_margin > action_horizon:
            raise ValueError("replan_margin must be between 0 and action_horizon.")
        return replan_margin
