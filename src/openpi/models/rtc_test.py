"""Tests for real-time chunking (RTC): prefix weights and guidance.

The guidance sign is the single easiest thing to get wrong when porting RTC into pi0, because
pi0's flow-matching time convention is reversed relative to the reference implementation. A
flipped sign still runs, still produces plausible-looking actions, and pushes the new chunk AWAY
from the previous one -- i.e. it makes the discontinuity worse rather than better, silently. So
these tests drive the real `guided_velocity` through the same Euler loop `sample_actions` uses,
with a synthetic velocity field standing in for the model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0

ACTION_HORIZON = 8
ACTION_DIM = 4


def _sample(velocity_fn, noise, *, num_steps=10, prev_action_chunk=None, prefix_weights=None, max_guidance_weight=5.0):
    """Replica of Pi0.sample_actions' Euler loop, minus the transformer."""
    dt = -1.0 / num_steps

    def step(carry):
        x_t, time = carry
        if prev_action_chunk is None:
            v_t = velocity_fn(x_t, time)
        else:
            v_t = pi0.guided_velocity(
                velocity_fn, x_t, time, prev_action_chunk, prefix_weights, max_guidance_weight
            )
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def _velocity_toward(mean):
    """Exact flow-matching velocity for data ~ N(`mean`, I), in this file's time convention.

    NOT a field that collapses onto a single point: for such a field the data prediction
    `a_hat = x_t - t*v` is constant in `x_t`, so its Jacobian is zero and guidance -- which is a
    VJP through exactly that map -- silently does nothing. Any test built on one would pass with
    the guidance sign flipped, or with guidance deleted outright.

    With x_t = t*eps + (1-t)*a and a ~ N(mean, I), the posterior mean is
    a_hat = mean + c(t) * (x_t - (1-t)*mean),  c(t) = (1-t) / (t^2 + (1-t)^2),
    and the velocity is v = (x_t - a_hat) / t.
    """

    def velocity_fn(x_t, time):
        c = (1.0 - time) / (time**2 + (1.0 - time) ** 2)
        a_hat = mean + c * (x_t - (1.0 - time) * mean)
        return (x_t - a_hat) / jnp.maximum(time, 1e-6)

    return velocity_fn


class TestPrefixWeights:
    def test_linear_schedule_shape(self):
        w = _model.get_prefix_weights(2, 5, 8, "linear")
        np.testing.assert_allclose(w, [1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0], atol=1e-6)

    def test_frozen_prefix_is_exactly_one(self):
        # The first `start` actions will already have been executed by the time the new chunk
        # takes over, so they must be fully pinned under every schedule that guides at all.
        for schedule in ("linear", "exp", "zeros", "ones"):
            w = _model.get_prefix_weights(3, 6, 8, schedule)
            np.testing.assert_allclose(w[:3], 1.0, atol=1e-6, err_msg=schedule)

    def test_zero_beyond_horizon(self):
        # Past `end` there is no valid overlap with the previous chunk -- guiding there would
        # pull the new chunk toward zero-padding.
        for schedule in ("linear", "exp", "zeros", "ones"):
            w = _model.get_prefix_weights(2, 5, 8, schedule)
            np.testing.assert_allclose(w[5:], 0.0, atol=1e-6, err_msg=schedule)

    def test_weights_in_unit_interval_and_nonincreasing(self):
        for schedule in ("linear", "exp"):
            w = _model.get_prefix_weights(2, 6, 10, schedule)
            assert np.all((w >= 0.0) & (w <= 1.0)), schedule
            assert np.all(np.diff(w) <= 1e-6), schedule

    def test_start_clamped_to_end(self):
        # A delay longer than the horizon must not produce negative-width decay / NaNs.
        w = _model.get_prefix_weights(9, 4, 8, "exp")
        assert np.all(np.isfinite(w))
        np.testing.assert_allclose(w[:4], 1.0, atol=1e-6)
        np.testing.assert_allclose(w[4:], 0.0, atol=1e-6)

    def test_rejects_unknown_schedule(self):
        with pytest.raises(ValueError, match="Invalid prefix attention schedule"):
            _model.get_prefix_weights(1, 2, 4, "quadratic")


class TestGuidedVelocity:
    """Guidance must pull the sampled chunk TOWARD the previous chunk where weighted."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.noise = jnp.asarray(rng.normal(size=(1, ACTION_HORIZON, ACTION_DIM)), dtype=jnp.float32)
        mean = jnp.asarray(rng.normal(size=(1, ACTION_HORIZON, ACTION_DIM)), dtype=jnp.float32)
        # The chunk we want to be pulled toward -- deliberately far from where the field lands.
        self.prev = jnp.asarray(rng.normal(size=(1, ACTION_HORIZON, ACTION_DIM)) + 3.0, dtype=jnp.float32)
        self.velocity_fn = _velocity_toward(mean)
        # Measured, not assumed: the baseline this field produces with no guidance at all.
        self.unguided = _sample(self.velocity_fn, self.noise)

    def test_harness_has_a_nonzero_denoiser_jacobian(self):
        # Guards the trap above: if a_hat did not depend on x_t, every guidance test here would
        # pass vacuously.
        jac = jax.jacobian(lambda x: x - 0.5 * self.velocity_fn(x, jnp.float32(0.5)))(self.noise)
        assert float(jnp.abs(jac).max()) > 1e-3

    def test_guidance_moves_pinned_actions_toward_prev_chunk(self):
        pinned, horizon = 3, 6
        weights = jnp.asarray(_model.get_prefix_weights(pinned, horizon, ACTION_HORIZON, "exp"))
        guided = _sample(self.velocity_fn, self.noise, prev_action_chunk=self.prev, prefix_weights=weights)

        err_guided = float(jnp.abs(guided[:, :pinned] - self.prev[:, :pinned]).mean())
        err_unguided = float(jnp.abs(self.unguided[:, :pinned] - self.prev[:, :pinned]).mean())

        # The decisive assertion: with the sign flipped, err_guided comes out LARGER than
        # err_unguided instead of a small fraction of it.
        assert err_guided < 0.25 * err_unguided, (
            f"guided error {err_guided:.4f} vs unguided {err_unguided:.4f} -- "
            "guidance is not pulling toward the previous chunk (check the sign convention)"
        )

    def test_guidance_does_not_disturb_unweighted_actions(self):
        # The velocity field here is elementwise, so zero-weighted indices must be untouched:
        # RTC constrains the start of the chunk and leaves the tail free to react to new obs.
        pinned, horizon = 2, 5
        weights = jnp.asarray(_model.get_prefix_weights(pinned, horizon, ACTION_HORIZON, "linear"))
        guided = _sample(self.velocity_fn, self.noise, prev_action_chunk=self.prev, prefix_weights=weights)
        np.testing.assert_allclose(guided[:, horizon:], self.unguided[:, horizon:], atol=1e-4)

    def test_stronger_guidance_pins_harder(self):
        pinned, horizon = 4, 6
        weights = jnp.asarray(_model.get_prefix_weights(pinned, horizon, ACTION_HORIZON, "ones"))
        errs = []
        for max_w in (1.0, 5.0, 20.0):
            out = _sample(
                self.velocity_fn,
                self.noise,
                prev_action_chunk=self.prev,
                prefix_weights=weights,
                max_guidance_weight=max_w,
            )
            errs.append(float(jnp.abs(out[:, :pinned] - self.prev[:, :pinned]).mean()))
        assert errs[0] > errs[1] > errs[2], f"error should shrink with max_guidance_weight, got {errs}"

    def test_zero_weights_reproduce_unguided_sampling(self):
        weights = jnp.zeros(ACTION_HORIZON, dtype=jnp.float32)
        guided = _sample(self.velocity_fn, self.noise, prev_action_chunk=self.prev, prefix_weights=weights)
        np.testing.assert_allclose(guided, self.unguided, atol=1e-4)

    def test_guidance_is_finite_across_the_whole_trajectory(self):
        # t*(1-t) hits zero at both ends of integration; the clip must keep this finite.
        weights = jnp.ones(ACTION_HORIZON, dtype=jnp.float32)
        for time in (1.0, 0.9, 0.5, 0.1, 1e-6):
            v = pi0.guided_velocity(
                self.velocity_fn, self.noise, jnp.float32(time), self.prev, weights, 5.0
            )
            assert bool(jnp.all(jnp.isfinite(v))), f"non-finite guided velocity at t={time}"

    def test_batch_elements_stay_independent(self):
        # The VJP is taken through a batched function with a batched cotangent; that is only
        # equal to the per-sample correction because attention never crosses the batch.
        rng = np.random.default_rng(7)
        noise = jnp.asarray(rng.normal(size=(3, ACTION_HORIZON, ACTION_DIM)), dtype=jnp.float32)
        target = jnp.asarray(rng.normal(size=(3, ACTION_HORIZON, ACTION_DIM)), dtype=jnp.float32)
        prev = jnp.asarray(rng.normal(size=(3, ACTION_HORIZON, ACTION_DIM)), dtype=jnp.float32)
        weights = jnp.asarray(_model.get_prefix_weights(3, 6, ACTION_HORIZON, "exp"))

        batched = _sample(_velocity_toward(target), noise, prev_action_chunk=prev, prefix_weights=weights)
        for i in range(3):
            single = _sample(
                _velocity_toward(target[i : i + 1]),
                noise[i : i + 1],
                prev_action_chunk=prev[i : i + 1],
                prefix_weights=weights,
            )
            np.testing.assert_allclose(batched[i : i + 1], single, atol=1e-4, err_msg=f"batch element {i}")

    def test_end_to_end_through_the_real_model(self):
        """Run guidance through an actual Pi0, not a stand-in velocity field.

        The synthetic tests above cannot catch the two ways this breaks in production: reverse-mode
        differentiation through the suffix pass may simply not work inside `lax.while_loop` under
        `jit` (the prefix KV cache is closed over, not differentiated), and `module_jit` has to
        accept the new keyword arguments. Both are all-or-nothing failures, so a tiny model with
        random weights and few denoising steps is enough to prove them out.

        Uses the "dummy" gemma variant deliberately: guidance holds the suffix activations for the
        backward pass, so running this at full 2B scale roughly doubles peak memory and OOMs the
        GPU when it lands after the other full-size model tests.
        """
        from openpi.models import pi0_config
        from openpi.shared import nnx_utils

        key = jax.random.key(0)
        config = pi0_config.Pi0Config(
            paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=8
        )
        model = config.create(key)
        obs = config.fake_obs(1)
        sample = nnx_utils.module_jit(model.sample_actions)

        noise = jax.random.normal(jax.random.key(1), (1, model.action_horizon, model.action_dim))
        prev = jax.random.normal(jax.random.key(2), (1, model.action_horizon, model.action_dim)) + 2.0
        weights = jnp.asarray(_model.get_prefix_weights(2, 5, model.action_horizon, "exp"))

        unguided = sample(key, obs, num_steps=4, noise=noise)
        guided = sample(
            key, obs, num_steps=4, noise=noise, prev_action_chunk=prev, prefix_weights=weights
        )

        assert guided.shape == unguided.shape == (1, model.action_horizon, model.action_dim)
        assert bool(jnp.all(jnp.isfinite(guided)))

        pinned = slice(None), slice(0, 2)
        err_guided = float(jnp.abs(guided[pinned] - prev[pinned]).mean())
        err_unguided = float(jnp.abs(unguided[pinned] - prev[pinned]).mean())
        assert err_guided < err_unguided, (
            f"guided error {err_guided:.4f} vs unguided {err_unguided:.4f} -- guidance had no "
            "effect (or the wrong sign) on the real model"
        )
        # The tail is NOT expected to match `unguided` here: unlike the diagonal test field, the
        # transformer couples action indices through attention, so constraining the prefix
        # legitimately moves the rest of the chunk. That coupling is the point -- the new chunk
        # replans around what it is committed to.

    def test_requires_both_prev_chunk_and_weights(self):
        # Guarding this in sample_actions matters: silently ignoring one of the two would mean
        # running plain sampling while the operator believes RTC is on.
        import inspect

        src = inspect.getsource(pi0.Pi0.sample_actions)
        assert "must be provided together" in src
