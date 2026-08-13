"""Tests for the real-time chunking broker's timing and conditioning.

The bugs these guard against are all silent at the API level -- the broker keeps returning
plausible actions while the robot jerks, freezes, or conditions on the wrong plan -- so each test
pins down an observable consequence (which chunk row came out, what went on the wire) rather than
just "it returned something".
"""

import concurrent.futures
import threading
import time

import numpy as np
import pytest

from openpi_client import action_chunk_broker
from openpi_client.action_chunk_broker import RTC_ENABLED
from openpi_client.action_chunk_broker import RTC_INFERENCE_DELAY
from openpi_client.action_chunk_broker import RTC_PREFIX_ATTENTION_HORIZON
from openpi_client.action_chunk_broker import RTC_PREV_CHUNK
from openpi_client.action_chunk_broker import RTC_RAW_ACTIONS

ACTION_DIM = 2
MODEL_DIM = 4
OBS = {"observation/state": np.zeros(ACTION_DIM, dtype=np.float32)}


class FakePolicy:
    """A policy whose chunks identify themselves, and whose latency the test controls.

    Chunk `c` has row `i` equal to `[100*c + i, ...]`, so an action alone says both which
    inference produced it and which index within that chunk it came from.
    """

    def __init__(self, horizon: int = 16, *, omit_raw: bool = False, fail_on: set[int] | None = None):
        self.horizon = horizon
        self.omit_raw = omit_raw
        self.fail_on = fail_on or set()
        self.requests: list[dict] = []
        self.gate: threading.Event | None = None
        self._counter = 0
        self._lock = threading.Lock()

    def infer(self, obs, noise=None):
        with self._lock:
            self._counter += 1
            call = self._counter
            self.requests.append({**obs, "_noise": noise})
        gate = self.gate
        if gate is not None and call > 1:  # never gate the cold-start call
            assert gate.wait(timeout=5.0), "test gate was never released"
        if call in self.fail_on:
            raise RuntimeError(f"simulated inference failure on call {call}")
        idx = np.arange(self.horizon)[:, None]
        results = {"actions": (100 * call + idx) * np.ones((1, ACTION_DIM), dtype=np.float32)}
        if not self.omit_raw:
            results[RTC_RAW_ACTIONS] = (100 * call + idx) * np.ones((1, MODEL_DIM), dtype=np.float32)
        return results

    def reset(self):
        pass


def _chunk_of(action: np.ndarray) -> int:
    return int(action[0]) // 100


def _index_of(action: np.ndarray) -> int:
    return int(action[0]) % 100


def _make(policy, **kwargs):
    kwargs.setdefault("execute_horizon", 4)
    kwargs.setdefault("inference_delay", 2)
    return action_chunk_broker.RealTimeActionChunkBroker(policy, **kwargs)


def _drive(broker, n: int) -> list[np.ndarray]:
    return [broker.infer(OBS)["actions"] for _ in range(n)]


def _wait_for_requests(policy, count: int, timeout: float = 5.0) -> None:
    """Block until the background thread has actually issued `count` requests.

    `_maybe_prefetch` only submits to an executor, so asserting on `policy.requests` right after
    an `infer()` call races the worker thread.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(policy.requests) >= count:
            return
        time.sleep(0.005)
    raise AssertionError(f"expected {count} requests, saw {len(policy.requests)}")


def _wait_for_pending(broker, timeout: float = 5.0) -> None:
    """Block until the in-flight request has completed (but not yet been installed)."""
    pending = broker._pending  # noqa: SLF001 - timing is the thing under test
    assert pending is not None, "expected a background inference to be in flight"
    concurrent.futures.wait([pending.future], timeout=timeout)
    assert pending.future.done(), "background inference did not complete in time"


class TestShiftChunk:
    def test_shifts_and_zero_pads(self):
        chunk = np.arange(5, dtype=np.float32)[:, None]
        np.testing.assert_allclose(
            action_chunk_broker._shift_chunk(chunk, 2)[:, 0], [2.0, 3.0, 4.0, 0.0, 0.0]
        )

    def test_zero_shift_is_identity(self):
        chunk = np.arange(5, dtype=np.float32)[:, None]
        np.testing.assert_allclose(action_chunk_broker._shift_chunk(chunk, 0), chunk)

    def test_full_shift_is_all_zeros(self):
        chunk = np.arange(5, dtype=np.float32)[:, None]
        np.testing.assert_allclose(action_chunk_broker._shift_chunk(chunk, 9), np.zeros_like(chunk))


class TestDispensing:
    def test_first_call_returns_first_action_of_first_chunk(self):
        broker = _make(FakePolicy())
        action = broker.infer(OBS)["actions"]
        assert (_chunk_of(action), _index_of(action)) == (1, 0)

    def test_actions_advance_one_index_per_call(self):
        broker = _make(FakePolicy())
        actions = _drive(broker, 4)
        assert [_index_of(a) for a in actions] == [0, 1, 2, 3]
        assert {_chunk_of(a) for a in actions} == {1}

    def test_noise_is_forwarded(self):
        policy = FakePolicy()
        broker = _make(policy)
        broker.infer(OBS, 0.25)
        assert policy.requests[0]["_noise"] == 0.25


class TestConditioning:
    def test_first_request_enables_rtc_but_has_no_previous_chunk(self):
        policy = FakePolicy()
        broker = _make(policy)
        broker.infer(OBS)
        assert policy.requests[0][RTC_ENABLED] is True
        assert RTC_PREV_CHUNK not in policy.requests[0]

    def test_prefetch_fires_one_execute_horizon_after_the_previous_one(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4)
        _drive(broker, 4)  # steps 0..3
        assert len(policy.requests) == 1, "must not replan before the execute horizon elapses"
        broker.infer(OBS)  # step 4 -> fires
        _wait_for_requests(policy, 2)
        policy.gate.set()

    def test_previous_chunk_is_time_shifted_to_the_moment_of_the_request(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4, inference_delay=2)
        _drive(broker, 5)  # the 5th call (step 4) fires the prefetch
        _wait_for_requests(policy, 2)
        policy.gate.set()

        request = policy.requests[1]
        # Chunk 1's raw rows are 100..115; shifted by 4 the request must start at 104, so the
        # server pins the new chunk's index 0 to the action for *now*, not for 4 steps ago.
        prev = request[RTC_PREV_CHUNK]
        assert prev.shape == (16, MODEL_DIM)
        np.testing.assert_allclose(prev[:, 0], list(range(104, 116)) + [0.0] * 4)
        assert request[RTC_INFERENCE_DELAY] == 2
        # Guidance must reach zero before the zero-padded tail begins.
        assert request[RTC_PREFIX_ATTENTION_HORIZON] == 12

    def test_prefix_horizon_never_covers_the_zero_padding(self):
        policy = FakePolicy(horizon=8)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=6, inference_delay=1)
        _drive(broker, 7)
        _wait_for_requests(policy, 2)
        policy.gate.set()
        request = policy.requests[1]
        prev = request[RTC_PREV_CHUNK]
        horizon = request[RTC_PREFIX_ATTENTION_HORIZON]
        assert horizon == 2
        np.testing.assert_allclose(prev[horizon:, 0], 0.0)


class TestSplicing:
    """The core fix: a chunk is a plan on an absolute time base, not a list to replay from 0."""

    def test_new_chunk_resumes_at_the_elapsed_index(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4, inference_delay=2)

        _drive(broker, 5)  # steps 0..4; the request fired at step 4 with origin 4
        policy.gate.set()
        _wait_for_pending(broker)

        action = broker.infer(OBS)["actions"]  # step 5
        assert _chunk_of(action) == 2, "should have swapped to the freshly computed chunk"
        # Step 5 is one step after the request's origin, so index 1 -- NOT index 0, which would
        # command an action aimed at where the arm was a step ago.
        assert _index_of(action) == 1

    def test_actions_stay_monotonic_across_the_splice(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4, inference_delay=2)

        indices = [_index_of(a) for a in _drive(broker, 5)]
        policy.gate.set()
        _wait_for_pending(broker)
        indices += [_index_of(broker.infer(OBS)["actions"]) for _ in range(3)]

        # Index within the (new) chunk restarts, but the underlying time base must not go
        # backwards: each action is one control step after the last.
        assert indices == [0, 1, 2, 3, 4, 1, 2, 3]

    def test_late_arrival_is_spliced_further_in_rather_than_rewound(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4, inference_delay=1)

        _drive(broker, 5)  # request fired at step 4, delay claims 1 step
        _drive(broker, 3)  # ... but 3 more steps pass before it lands (steps 5,6,7)
        policy.gate.set()
        _wait_for_pending(broker)

        action = broker.infer(OBS)["actions"]  # step 8
        assert _chunk_of(action) == 2
        assert _index_of(action) == 4, "must resume at elapsed steps since origin, not at 0"

    def test_fully_expired_chunk_is_discarded(self):
        policy = FakePolicy(horizon=6)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=3, inference_delay=1)

        _drive(broker, 3)  # step 2 is the last before the fire at step 3
        _drive(broker, 3)  # steps 3,4,5 -- request fired at step 3 with origin 3
        # Hold long enough that the chunk fired at origin 3 is useless by the time it lands: at
        # step 15 its index would be 12, well past its 6-action horizon.
        _drive(broker, 9)
        policy.gate.set()
        _wait_for_pending(broker)

        action = broker.infer(OBS)["actions"]
        # The expired chunk must be dropped and replaced by a fresh synchronous inference, rather
        # than indexed past its end (IndexError) or restarted at 0 (a jump back in time).
        assert _chunk_of(action) == 3
        assert _index_of(action) == 0


class TestExhaustion:
    def test_holds_last_action_while_inference_is_in_flight(self):
        policy = FakePolicy(horizon=6)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=3, inference_delay=1, exhausted_behavior="hold")

        actions = _drive(broker, 6)  # consumes chunk 1 entirely; request fired at step 3
        held = broker.infer(OBS)["actions"]  # step 6: nothing to dispense
        np.testing.assert_allclose(held, actions[-1])
        policy.gate.set()

    def test_recovers_after_holding(self):
        policy = FakePolicy(horizon=6)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=3, inference_delay=1, exhausted_behavior="hold")

        _drive(broker, 6)
        broker.infer(OBS)  # holds
        policy.gate.set()
        _wait_for_pending(broker)
        action = broker.infer(OBS)["actions"]
        assert _chunk_of(action) == 2, "must resume from the new chunk once it lands"

    def test_does_not_freeze_forever_when_nothing_is_in_flight(self):
        """Regression: the old broker held the last action for the rest of the episode.

        With no request in flight and `hold` behavior it returned `_last_result` unconditionally
        and never inferred again -- the arm stopped, permanently, with no error.
        """
        policy = FakePolicy(horizon=6)
        # execute_horizon == horizon, so the chunk runs out before any prefetch is ever fired.
        broker = _make(policy, execute_horizon=6, inference_delay=0, exhausted_behavior="hold")

        first = _drive(broker, 6)
        assert len(policy.requests) == 1
        action = broker.infer(OBS)["actions"]

        assert _chunk_of(action) == 2, "broker froze instead of issuing a synchronous inference"
        assert not np.allclose(action, first[-1])

    def test_blocking_behavior_waits_for_the_chunk(self):
        policy = FakePolicy(horizon=6)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=3, inference_delay=1, exhausted_behavior="block")

        _drive(broker, 6)
        released = threading.Timer(0.05, policy.gate.set)
        released.start()
        action = broker.infer(OBS)["actions"]  # must block rather than hold
        released.join()
        assert _chunk_of(action) == 2


class TestRobustness:
    def test_reset_discards_a_chunk_computed_before_it(self):
        policy = FakePolicy(horizon=16)
        policy.gate = threading.Event()
        broker = _make(policy, execute_horizon=4, inference_delay=2)

        _drive(broker, 5)  # fires a request against the pre-reset observation
        broker.reset()
        policy.gate.set()

        action = broker.infer(OBS)["actions"]
        # Chunk 2 was aimed at the pre-intervention pose; after a handback it must not be used.
        assert _chunk_of(action) == 3
        assert _index_of(action) == 0

    def test_background_failure_does_not_reach_the_control_loop(self):
        policy = FakePolicy(horizon=6, fail_on={2})
        broker = _make(policy, execute_horizon=3, inference_delay=1, exhausted_behavior="hold")

        _drive(broker, 3)
        broker.infer(OBS)  # step 3 fires the doomed background request
        actions = _drive(broker, 2)  # steps 4,5 finish the current chunk
        action = broker.infer(OBS)["actions"]  # step 6: falls back to a synchronous call

        assert _chunk_of(action) == 3
        assert all(np.all(np.isfinite(a)) for a in actions)

    def test_missing_raw_actions_gives_an_actionable_error(self):
        broker = _make(FakePolicy(omit_raw=True))
        with pytest.raises(ValueError, match=RTC_RAW_ACTIONS):
            broker.infer(OBS)

    def test_rejects_delay_longer_than_execute_horizon(self):
        with pytest.raises(ValueError, match="inference_delay"):
            _make(FakePolicy(), execute_horizon=2, inference_delay=3)

    def test_rejects_timing_budget_that_overruns_the_action_horizon(self):
        # Each chunk is executed over [delay, delay + execute_horizon); 6 + 3 > 8 runs off the end.
        broker = _make(FakePolicy(horizon=8), execute_horizon=6, inference_delay=3)
        with pytest.raises(ValueError, match="action horizon"):
            broker.infer(OBS)

    def test_rejects_unknown_schedule(self):
        with pytest.raises(ValueError, match="prefix_attention_schedule"):
            _make(FakePolicy(), prefix_attention_schedule="quadratic")


class TestPlainBroker:
    """The non-RTC broker is untouched and must keep working against plain servers."""

    def test_dispenses_a_chunk_then_re_infers(self):
        policy = FakePolicy(horizon=4)
        broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=4)
        chunks = [_chunk_of(broker.infer(OBS)["actions"]) for _ in range(5)]
        assert chunks == [1, 1, 1, 1, 2]
