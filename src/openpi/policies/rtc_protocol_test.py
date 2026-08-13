"""Tests for the real-time chunking wire protocol between the robot client and the policy server.

Everything here is about failures that are silent rather than loud. The RTC control fields ride
inside the observation dict, and the input transforms drop keys they do not recognize -- so a
constant renamed on one side does not raise, it just turns guidance off and leaves the operator
with a policy that looks like it is running RTC and is not.
"""

import time

import numpy as np
from openpi_client import action_chunk_broker as _broker
import pytest

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server as _server


class _StubModel:
    action_horizon = 10
    action_dim = 32


def _stub_policy(*, is_pytorch: bool = False) -> _policy.Policy:
    """A Policy shell with just enough state for the RTC plumbing, and no checkpoint to load."""
    policy = object.__new__(_policy.Policy)
    policy._model = _StubModel()  # noqa: SLF001
    policy._is_pytorch_model = is_pytorch  # noqa: SLF001
    return policy


def _rtc_request(**overrides):
    request = {
        _policy.RTC_ENABLED: True,
        _policy.RTC_PREV_CHUNK: np.zeros((10, 32), dtype=np.float32),
        _policy.RTC_INFERENCE_DELAY: 2,
        _policy.RTC_PREFIX_ATTENTION_HORIZON: 6,
    }
    request.update(overrides)
    return request


class TestConstantParity:
    def test_client_and_server_agree_on_every_field_name(self):
        pairs = [
            (_broker.RTC_ENABLED, _policy.RTC_ENABLED),
            (_broker.RTC_PREV_CHUNK, _policy.RTC_PREV_CHUNK),
            (_broker.RTC_INFERENCE_DELAY, _policy.RTC_INFERENCE_DELAY),
            (_broker.RTC_PREFIX_ATTENTION_HORIZON, _policy.RTC_PREFIX_ATTENTION_HORIZON),
            (_broker.RTC_PREFIX_ATTENTION_SCHEDULE, _policy.RTC_PREFIX_ATTENTION_SCHEDULE),
            (_broker.RTC_MAX_GUIDANCE_WEIGHT, _policy.RTC_MAX_GUIDANCE_WEIGHT),
            (_broker.RTC_RAW_ACTIONS, _policy.RTC_RAW_ACTIONS),
        ]
        for client_name, server_name in pairs:
            assert client_name == server_name

    def test_server_pops_every_field_the_client_can_send(self):
        # Any control field left in the dict reaches the input transforms as if it were an
        # observation.
        client_fields = {
            _broker.RTC_ENABLED,
            _broker.RTC_PREV_CHUNK,
            _broker.RTC_INFERENCE_DELAY,
            _broker.RTC_PREFIX_ATTENTION_HORIZON,
            _broker.RTC_PREFIX_ATTENTION_SCHEDULE,
            _broker.RTC_MAX_GUIDANCE_WEIGHT,
        }
        assert client_fields == set(_policy._RTC_FIELDS)  # noqa: SLF001


class TestPopRtcControl:
    def test_strips_control_fields_from_the_observation(self):
        inputs = {"observation/state": np.zeros(14), **_rtc_request()}
        control = _policy._pop_rtc_control(inputs)  # noqa: SLF001
        assert control is not None
        assert set(inputs) == {"observation/state"}

    def test_returns_none_when_rtc_is_off(self):
        inputs = {"observation/state": np.zeros(14)}
        assert _policy._pop_rtc_control(inputs) is None  # noqa: SLF001

    def test_strips_control_fields_even_when_disabled(self):
        # Otherwise a client that sets enabled=False would poison the transforms instead of
        # simply getting unguided sampling.
        inputs = {"observation/state": np.zeros(14), **_rtc_request(**{_policy.RTC_ENABLED: False})}
        assert _policy._pop_rtc_control(inputs) is None  # noqa: SLF001
        assert set(inputs) == {"observation/state"}


class TestSampleKwargs:
    def test_first_call_without_a_previous_chunk_samples_unguided(self):
        control = {_policy.RTC_ENABLED: True}
        assert _stub_policy()._rtc_sample_kwargs(control) == {}  # noqa: SLF001

    def test_builds_guidance_arguments(self):
        kwargs = _stub_policy()._rtc_sample_kwargs(_rtc_request())  # noqa: SLF001
        assert kwargs["prev_action_chunk"].shape == (1, 10, 32), "model expects a batch dimension"
        assert kwargs["prefix_weights"].shape == (10,)
        assert kwargs["max_guidance_weight"] == 5.0
        # Both are required together by sample_actions.
        assert {"prev_action_chunk", "prefix_weights"} <= set(kwargs)

    def test_weights_match_the_requested_schedule(self):
        kwargs = _stub_policy()._rtc_sample_kwargs(  # noqa: SLF001
            _rtc_request(**{_policy.RTC_PREFIX_ATTENTION_SCHEDULE: "linear"})
        )
        expected = _model.get_prefix_weights(2, 6, 10, "linear")
        np.testing.assert_allclose(np.asarray(kwargs["prefix_weights"]), expected, atol=1e-6)

    def test_rejects_a_chunk_in_robot_space(self):
        # The classic mistake: echoing back the un-normalized 14-DOF actions instead of the raw
        # model-space chunk. Broadcasting would otherwise turn this into a shape error deep inside
        # the sampler, or worse, silently succeed.
        with pytest.raises(ValueError, match="MODEL action space"):
            _stub_policy()._rtc_sample_kwargs(  # noqa: SLF001
                _rtc_request(**{_policy.RTC_PREV_CHUNK: np.zeros((10, 14), dtype=np.float32)})
            )

    def test_rejects_a_chunk_with_the_wrong_horizon(self):
        with pytest.raises(ValueError, match="MODEL action space"):
            _stub_policy()._rtc_sample_kwargs(  # noqa: SLF001
                _rtc_request(**{_policy.RTC_PREV_CHUNK: np.zeros((4, 32), dtype=np.float32)})
            )

    def test_rejects_out_of_range_prefix_horizon(self):
        with pytest.raises(ValueError, match="prefix_attention_horizon"):
            _stub_policy()._rtc_sample_kwargs(  # noqa: SLF001
                _rtc_request(**{_policy.RTC_PREFIX_ATTENTION_HORIZON: 99})
            )

    def test_rejects_negative_delay(self):
        with pytest.raises(ValueError, match="inference_delay"):
            _stub_policy()._rtc_sample_kwargs(_rtc_request(**{_policy.RTC_INFERENCE_DELAY: -1}))  # noqa: SLF001

    def test_pytorch_models_are_rejected_rather_than_silently_unguided(self):
        with pytest.raises(NotImplementedError, match="JAX"):
            _stub_policy(is_pytorch=True)._rtc_sample_kwargs(_rtc_request())  # noqa: SLF001


class _EchoPolicy:
    """A policy that records what actually arrived over the wire."""

    def __init__(self, horizon: int = 10, action_dim: int = 32):
        self.horizon = horizon
        self.action_dim = action_dim
        self.requests: list[dict] = []

    def infer(self, obs, *, noise=None):
        self.requests.append(obs)
        return {
            "actions": np.zeros((self.horizon, 14), dtype=np.float32),
            _policy.RTC_RAW_ACTIONS: np.ones((self.horizon, self.action_dim), dtype=np.float32),
        }

    def reset(self):
        pass


@pytest.fixture
def rtc_server():
    """A real WebsocketPolicyServer on a free port, backed by `_EchoPolicy`."""
    import socket
    import threading

    from openpi_client import websocket_client_policy

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    policy = _EchoPolicy()
    server = _server.WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # The client retries a refused connection, so it covers the startup race for us.
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)
    return policy, client


class TestOverTheWire:
    """End-to-end through msgpack and the websocket, with this repo's own client.

    This is the pairing that used to be broken: the client wrapped requests in an envelope the
    server never opened, so it could not talk to its own server at all. These tests fail loudly if
    that regresses, and they cover msgpack's handling of the non-array RTC fields (a bool, ints, a
    string, a float), which would otherwise only ever be exercised on a robot.
    """

    def test_client_and_server_can_talk_at_all(self, rtc_server):
        policy, client = rtc_server
        result = client.infer({"observation/state": np.zeros(14, dtype=np.float32)})
        assert result["actions"].shape == (10, 14)
        assert "observation/state" in policy.requests[0], "server received the envelope, not the obs"
        assert "method" not in policy.requests[0]

    def test_broker_round_trips_the_rtc_fields(self, rtc_server):
        from openpi_client import action_chunk_broker

        policy, client = rtc_server
        broker = action_chunk_broker.RealTimeActionChunkBroker(
            client,
            execute_horizon=4,
            inference_delay=2,
            prefix_attention_schedule="linear",
            max_guidance_weight=7.5,
        )
        obs = {"observation/state": np.zeros(14, dtype=np.float32)}
        for _ in range(5):  # the 5th call fires the conditioned request
            broker.infer(obs)
        # The conditioned request goes out on a background thread; wait for it to land server-side.
        deadline = time.monotonic() + 10.0
        while len(policy.requests) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        broker.close()

        assert len(policy.requests) >= 2, "broker never issued a second, conditioned request"
        request = policy.requests[1]
        assert request[_policy.RTC_ENABLED] is True
        assert request[_policy.RTC_INFERENCE_DELAY] == 2
        assert request[_policy.RTC_PREFIX_ATTENTION_HORIZON] == 6
        assert request[_policy.RTC_PREFIX_ATTENTION_SCHEDULE] == "linear"
        assert request[_policy.RTC_MAX_GUIDANCE_WEIGHT] == pytest.approx(7.5)
        prev = request[_policy.RTC_PREV_CHUNK]
        assert isinstance(prev, np.ndarray)
        assert prev.shape == (10, 32), "the previous chunk must survive msgpack as an array"

    def test_server_still_accepts_a_client_that_knows_nothing_about_rtc(self, rtc_server):
        policy, client = rtc_server
        client.infer({"observation/state": np.zeros(14, dtype=np.float32)})
        assert not any(key.startswith("rtc/") for key in policy.requests[0])


class TestRequestUnpacking:
    """The server must speak to both this repo's client and upstream openpi's."""

    def test_unwraps_this_repos_envelope(self):
        obs = {"observation/state": np.zeros(14)}
        method, unpacked = _server._unpack_request({"method": "infer", "obs": obs})  # noqa: SLF001
        assert method == "infer"
        assert unpacked is obs

    def test_accepts_a_bare_observation_dict(self):
        obs = {"observation/state": np.zeros(14), "prompt": "pick up the banana"}
        method, unpacked = _server._unpack_request(obs)  # noqa: SLF001
        assert method == "infer"
        assert unpacked is obs

    def test_routes_other_methods(self):
        method, _ = _server._unpack_request({"method": "get_prefix_rep", "obs": {}})  # noqa: SLF001
        assert method == "get_prefix_rep"

    def test_does_not_mistake_an_observation_named_obs_for_an_envelope(self):
        # A bare observation dict that happens to contain an "obs" key must not be unwrapped;
        # requiring the key set to be a subset of {method, obs} is what prevents that.
        obs = {"obs": np.zeros(3), "observation/state": np.zeros(14)}
        method, unpacked = _server._unpack_request(obs)  # noqa: SLF001
        assert method == "infer"
        assert unpacked is obs
