import threading
import time

import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _CountingPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.obs_steps = []

    def infer(self, obs, noise=None):
        self.obs_steps.append(int(obs["step"]))
        start = int(obs["step"]) * 10
        return {"actions": np.arange(start, start + 4, dtype=np.float32)[:, None]}


def test_realtime_broker_prefetches_before_chunk_exhaustion():
    policy = _CountingPolicy()
    broker = action_chunk_broker.RealTimeActionChunkBroker(
        policy=policy,
        action_horizon=4,
        replan_margin=2,
        exhausted_behavior="block",
    )

    outputs = [broker.infer({"step": step})["actions"].item() for step in range(5)]

    assert outputs == [0, 1, 2, 3, 20]
    assert policy.obs_steps == [0, 2]
    broker.close()


class _SlowSecondPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.call_count = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, obs, noise=None):
        call_count = self.call_count
        self.call_count += 1
        if call_count == 1:
            self.started.set()
            self.release.wait(timeout=5)
        start = call_count * 10
        return {"actions": np.arange(start, start + 2, dtype=np.float32)[:, None]}


def test_realtime_broker_holds_last_action_when_prefetch_overruns():
    policy = _SlowSecondPolicy()
    broker = action_chunk_broker.RealTimeActionChunkBroker(
        policy=policy,
        action_horizon=2,
        replan_margin=1,
        exhausted_behavior="hold",
    )

    assert broker.infer({"step": 0})["actions"].item() == 0
    assert broker.infer({"step": 1})["actions"].item() == 1
    assert policy.started.wait(timeout=1)

    t0 = time.perf_counter()
    assert broker.infer({"step": 2})["actions"].item() == 1
    assert time.perf_counter() - t0 < 0.2

    policy.release.set()
    for _ in range(20):
        output = broker.infer({"step": 3})["actions"].item()
        if output == 10:
            break
        time.sleep(0.01)

    assert output == 10
    broker.close()
