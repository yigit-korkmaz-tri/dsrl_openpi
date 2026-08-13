from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Real-time chunking control fields. These travel in the observation dict because that is the only
# thing the websocket protocol carries, but they are NOT observations -- they are sampler settings,
# so they are popped off before the input transforms run (which would otherwise drop them silently).
RTC_ENABLED = "rtc/enabled"
RTC_PREV_CHUNK = "rtc/prev_chunk"
RTC_INFERENCE_DELAY = "rtc/inference_delay"
RTC_PREFIX_ATTENTION_HORIZON = "rtc/prefix_attention_horizon"
RTC_PREFIX_ATTENTION_SCHEDULE = "rtc/prefix_attention_schedule"
RTC_MAX_GUIDANCE_WEIGHT = "rtc/max_guidance_weight"
# Echoed back to the client: the action chunk in MODEL space (normalized, padded to action_dim),
# i.e. before the output transforms un-normalize and slice it down to the robot's DOF. The client
# stores it and hands it back as RTC_PREV_CHUNK on the next call.
#
# Why echo instead of letting the client send back the actions it received: guidance operates in
# model space, so a robot-space chunk would have to be re-normalized and re-padded server-side --
# an inverse of the output transform chain that does not exist and would have to be maintained
# per-policy. Round-tripping the raw chunk is exact and policy-agnostic; it costs ~1-4 KB.
RTC_RAW_ACTIONS = "rtc/raw_actions"

_RTC_FIELDS = (
    RTC_ENABLED,
    RTC_PREV_CHUNK,
    RTC_INFERENCE_DELAY,
    RTC_PREFIX_ATTENTION_HORIZON,
    RTC_PREFIX_ATTENTION_SCHEDULE,
    RTC_MAX_GUIDANCE_WEIGHT,
)


def _pop_rtc_control(inputs: dict) -> dict | None:
    """Remove the RTC control fields from `inputs`, returning them if RTC is on."""
    control = {key: inputs.pop(key) for key in _RTC_FIELDS if key in inputs}
    if not control.get(RTC_ENABLED):
        return None
    return control


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        rtc = _pop_rtc_control(inputs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if rtc is not None:
            sample_kwargs.update(self._rtc_sample_kwargs(rtc))

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Snapshot before the output transform, which un-normalizes and slices in place.
        raw_actions = np.array(outputs["actions"], copy=True) if rtc is not None else None

        outputs = self._output_transform(outputs)
        if raw_actions is not None:
            outputs[RTC_RAW_ACTIONS] = raw_actions
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _rtc_sample_kwargs(self, rtc: dict) -> dict[str, Any]:
        """Translate the client's RTC control fields into `sample_actions` kwargs.

        The schedule is resolved to a weight vector HERE, on the host, so that neither the
        schedule string nor the (frequently changing) integer delay ever reaches the jitted
        `sample_actions` -- see `model.get_prefix_weights`.
        """
        if self._is_pytorch_model:
            raise NotImplementedError(
                "Real-time chunking is implemented for the JAX models only; the PyTorch "
                "sample_actions does not accept prev_action_chunk/prefix_weights."
            )

        prev_chunk = rtc.get(RTC_PREV_CHUNK)
        if prev_chunk is None:
            # First call of an episode: there is no previous chunk to stay consistent with, so
            # this is ordinary unguided sampling. The client still gets RTC_RAW_ACTIONS back and
            # conditions on it from the next call onward.
            return {}

        action_horizon = self._model.action_horizon
        prev_chunk = np.asarray(prev_chunk, dtype=np.float32)
        expected = (action_horizon, self._model.action_dim)
        if prev_chunk.shape != expected:
            raise ValueError(
                f"{RTC_PREV_CHUNK} must be the previous chunk in MODEL action space with shape "
                f"{expected}, got {prev_chunk.shape}. Echo back the {RTC_RAW_ACTIONS} array from "
                "the previous response verbatim (time-shifted) -- not the un-normalized actions."
            )

        delay = int(rtc.get(RTC_INFERENCE_DELAY, 0))
        horizon = int(rtc.get(RTC_PREFIX_ATTENTION_HORIZON, action_horizon))
        if delay < 0:
            raise ValueError(f"{RTC_INFERENCE_DELAY} must be non-negative, got {delay}.")
        if not 0 <= horizon <= action_horizon:
            raise ValueError(
                f"{RTC_PREFIX_ATTENTION_HORIZON} must be in [0, {action_horizon}], got {horizon}."
            )

        weights = _model.get_prefix_weights(
            delay, horizon, action_horizon, rtc.get(RTC_PREFIX_ATTENTION_SCHEDULE, "exp")
        )
        return {
            "prev_action_chunk": jnp.asarray(prev_chunk)[np.newaxis, ...],
            "prefix_weights": jnp.asarray(weights),
            "max_guidance_weight": float(rtc.get(RTC_MAX_GUIDANCE_WEIGHT, 5.0)),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
