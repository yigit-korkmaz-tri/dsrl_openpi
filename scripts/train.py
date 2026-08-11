import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
            # Flow-MILE: snapshot the initial (collection-policy) weights as the frozen rollout policy
            # pi_0. Fixed for the whole run. None (and no memory cost) for non-Flow-MILE configs.
            rollout_params=params if config.flow_mile is not None else None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# =====================================================================================
# Flow-MILE (see openpi.training.config.FlowMileParams). Adds the MILE intervention-probit
# objective on top of pi0/pi0.5's flow-matching loss:
#   total = BC(labels {1,2})  +  lambda * BCE_probit(labels {0,1})
# with condition_intervention_on_action + condition_nonintervention_on_robot (no proximal / no
# score-gap norm). The score ell(a,s) is the (reference-relative) flow-matching loss under the online
# vs frozen rollout policy; masks (not boolean indexing) keep it JIT-friendly.
# =====================================================================================
_FLOW_MILE_EPS = 1e-4


def _normal_cdf(x: at.Array) -> at.Array:
    """Standard normal CDF Phi(x)."""
    return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))


def _tile_obs(observation: _model.Observation, k: int) -> _model.Observation:
    """Tile an Observation along the batch axis: row (j*B + b) -> obs b (matches actions.reshape(K*B))."""
    return jax.tree.map(lambda x: jnp.concatenate([x] * k, axis=0), observation)


def _flow_loss(model: _model.BaseModel, rng: at.KeyArrayLike, observation, actions, score_mc: int) -> at.Array:
    """Per-sample flow-matching loss ``[*b]`` = mean over ah of pi0 compute_loss, averaged over
    ``score_mc`` (t, x0) draws. Used as the (negative) log-prob proxy. train=False (deterministic)."""

    def one(r):
        return jnp.mean(model.compute_loss(r, observation, actions, train=False), axis=-1)

    if score_mc <= 1:
        return one(rng)
    return jnp.mean(jnp.stack([one(r) for r in jax.random.split(rng, score_mc)], axis=0), axis=0)


def _flow_mile_grads(config, online_model, state, rng, observation, actions, interventions, diff_state):
    """Compute the Flow-MILE loss + grads w.r.t. the online model's trainable params.

    Everything independent of the online params theta (all sampling; the frozen-policy reference
    flow-losses) is precomputed OUTSIDE value_and_grad as constants, so loss_fn runs only the online
    model and stays a single-module grad (like the standard path). Returns (loss, aux, grads).
    """
    fm = config.flow_mile
    k = fm.num_samples
    b = actions.shape[0]
    ah, ad = actions.shape[-2], actions.shape[-1]
    interventions = jnp.reshape(interventions, (b,)).astype(jnp.float32)

    rng_rs, rng_os, rng_r, rng_o, rng_l, rng_bc = jax.random.split(rng, 6)

    rollout_model = nnx.merge(state.model_def, state.rollout_params)  # frozen pi_0 (shares state arrays)
    rollout_model.eval()

    obs_k = _tile_obs(observation, k)  # [K*B, ...]
    # Sample K chunks per state from the frozen rollout policy and the online policy (stop-grad).
    rollout_samples = jax.lax.stop_gradient(
        rollout_model.sample_actions(
            rng_rs, obs_k, num_steps=fm.num_sample_steps, noise=jax.random.normal(rng_rs, (k * b, ah, ad))
        )
    )
    online_samples = jax.lax.stop_gradient(
        online_model.sample_actions(
            rng_os, obs_k, num_steps=fm.num_sample_steps, noise=jax.random.normal(rng_os, (k * b, ah, ad))
        )
    )
    # Frozen-policy (reference) flow losses -- constants (no theta dependence).
    ref_rollout = jax.lax.stop_gradient(_flow_loss(rollout_model, rng_r, obs_k, rollout_samples, fm.score_mc_samples))
    ref_online = jax.lax.stop_gradient(_flow_loss(rollout_model, rng_o, obs_k, online_samples, fm.score_mc_samples))
    ref_logged = jax.lax.stop_gradient(_flow_loss(rollout_model, rng_l, observation, actions, fm.score_mc_samples))

    ref_rel, w, beta, c, lam = (
        fm.reference_relative_score,
        fm.expected_rollout_score_weight,
        fm.probit_scale,
        fm.intervention_cost,
        fm.lambda_intervention,
    )

    def loss_fn(model, bc_rng):
        # Online (theta) flow losses -- SAME rng as the reference for paired (t, x0), so the
        # reference-relative score has low variance. Gradients flow through these.
        th_rollout = _flow_loss(model, rng_r, obs_k, rollout_samples, fm.score_mc_samples)  # [K*B]
        th_online = _flow_loss(model, rng_o, obs_k, online_samples, fm.score_mc_samples)  # [K*B]
        th_logged = _flow_loss(model, rng_l, observation, actions, fm.score_mc_samples)  # [B]

        if ref_rel:  # ell = flow_loss_0 - flow_loss_theta
            rollout_scores = (ref_rollout - th_rollout).reshape(k, b)
            training_scores = (ref_online - th_online).reshape(k, b)
            logged_scores = ref_logged - th_logged
        else:  # ell = -flow_loss_theta
            rollout_scores = (-th_rollout).reshape(k, b)
            training_scores = (-th_online).reshape(k, b)
            logged_scores = -th_logged

        expected_rollout_score = jnp.mean(rollout_scores, axis=0)  # [B] baseline E_{a0~pi0} ell(a0,s)
        # condition_nonintervention_on_robot: on label-0 rows the baseline is the LOGGED robot score.
        nonint = interventions < 0.5
        baseline = jnp.where(nonint, logged_scores, expected_rollout_score)  # [B]

        # Marginal p(nu=1|s) = E_{a~pi_theta} Phi(beta(ell(a,s) - w*baseline) - c)  (used for label-0).
        marginal_probs = jnp.mean(_normal_cdf(beta * (training_scores - w * baseline[None, :]) - c), axis=0)  # [B]
        # condition_intervention_on_action: label-1 uses the observed human action score vs the
        # SAMPLED rollout baseline: p(nu=1|s,a_h) = Phi(beta(ell(a_h,s) - w*E_rollout) - c).
        observed_probs = _normal_cdf(beta * (logged_scores - w * expected_rollout_score) - c)  # [B]
        label1 = (interventions > 0.5) & (interventions < 1.5)
        probs = jnp.clip(jnp.where(label1, observed_probs, marginal_probs), _FLOW_MILE_EPS, 1.0 - _FLOW_MILE_EPS)

        # Intervention BCE on labels {0,1}; label-2 (offline) excluded.
        online_mask = (interventions < 1.5).astype(jnp.float32)
        targets = jnp.clip(interventions, 0.0, 1.0)
        bce_per = -(targets * jnp.log(probs) + (1.0 - targets) * jnp.log(1.0 - probs))
        bce = jnp.sum(bce_per * online_mask) / jnp.clip(jnp.sum(online_mask), 1.0, None)

        # Flow-matching BC on labels {1,2} (human corrections + offline demos).
        action_mask = (interventions > 0.5).astype(jnp.float32)
        bc_per = jnp.mean(model.compute_loss(bc_rng, observation, actions, train=True), axis=-1)  # [B]
        bc = jnp.sum(bc_per * action_mask) / jnp.clip(jnp.sum(action_mask), 1.0, None)

        total = bc + lam * bce
        aux = {
            "bc_loss": bc,
            "bce_loss": bce,
            "intervention_prob_mean": jnp.mean(probs),
            "intervention_frac": jnp.mean(label1.astype(jnp.float32)),
            "expected_rollout_score_mean": jnp.mean(expected_rollout_score),
            "logged_score_mean": jnp.mean(logged_scores),
        }
        return total, aux

    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(online_model, rng_bc)
    return loss, aux, grads


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)

    if config.flow_mile is not None:
        observation, actions, interventions = batch
        loss, extra_info, grads = _flow_mile_grads(
            config, model, state, train_rng, observation, actions, interventions, diff_state
        )
    else:
        observation, actions = batch

        def loss_fn(model, rng, observation, actions):
            return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
        extra_info = {}

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **extra_info,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
