"""ModelBridge for a pi05 / openpi policy served over websocket — for `rd infer`.

Unlike :mod:`raiden.bridges.act_bridge`, this bridge holds **no model**: the openpi
`serve_policy.py` process owns the checkpoint and the GPU, and this bridge is a thin
client. That makes `rd infer --intervene` (HG-DAgger) work against pi05 with no changes
to the inference loop — the loop is model-agnostic, and the openpi action layout already
matches raiden's ModelBridge contract exactly.

Run (two processes on the robot host):

    # 1) serve the checkpoint (owns the GPU)
    cd ~/robometer-policy-learning && uv run --no-sync python \
        third_party/dsrl_openpi/scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yam_pickbanana --policy.dir=~/pi05_yam_pickbanana/6000

    # 2) drive the arm + collect corrections
    rd infer --bridge raiden.bridges.openpi_bridge:OpenPIBridge \
        --ckpt-path localhost:8000 \
        --bridge-kwargs '{"prompt": "pick up the banana"}' \
        --action-type joint --action-hz 15 --resize-images "" --no-depth \
        --intervene --dagger-task banana_dagger_r1

LAYOUT (verified against the code, 2026-08-06):
  * openpi state  (14,) = [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
  * openpi action (14,) = same layout, ABSOLUTE joint targets
  * raiden ModelBridge joint contract = [left(6), left_grip(1), right(6), right_grip(1)]
  => IDENTICAL. State is a plain concat of the two 7-D proprios (DOF=7 in server.py:85,
     6 revolute + 1 gripper), and the action is returned untouched. NO reorder, NO swap.

!!! IMAGES — the two things that are easy to get wrong
  [1] LETTERBOX, DON'T SQUASH. openpi's ``preprocess_observation``
      (dsrl_openpi/src/openpi/models/model.py:150-166) resizes to 224x224 with
      ``resize_with_pad`` — aspect preserved, zero-padded — and ONLY when the input is
      not already 224x224. Training fed 1280x720 frames, so the model learned
      LETTERBOXED images with black bars. Pre-squashing 16:9 -> 1:1 with cv2.resize
      makes resize_with_pad a no-op and hands the policy a vertically stretched scene.
      So we call ``resize_with_pad`` ourselves (same function openpi uses) and must be
      given the NATIVE frame: run `rd infer` with ``--resize-images ""`` so the server
      does not squash to its 384x384 default first.
  [2] DO NOT FLIP right_wrist_camera. It is rotated 180 deg at capture by
      ``server.py:_FLIP_CAMERAS`` AND by ``converter.py:_FLIP_CAMERAS`` on the SVO2 ->
      training path, so the training images already have it and raiden's obs already
      matches. Flipping again here would invert it.
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
from pathlib import Path
import sys
import threading

import numpy as np
from raiden.inference import ModelBridge

# openpi-client MUST come from a checkout whose websocket protocol matches the SERVER's.
# dsrl_openpi's tree is internally skewed: its client wraps the payload in an envelope
# {"method": "infer", "obs": obs}, but its websocket_policy_server hands the unpacked
# message STRAIGHT to policy.infer() — so the transform sees {"method", "obs"} and dies
# with KeyError: 'observation/image_head'. The upstream ~/openpi client sends the bare
# obs dict, which is what the server (byte-identical file in both trees) expects.
_OPENPI_CLIENT_HOME = Path.home() / "openpi" / "packages" / "openpi-client" / "src"
if str(_OPENPI_CLIENT_HOME) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_HOME))

MODEL_IMG_SIZE = 224

# raiden camera NAME -> openpi observation key. YAM names its cams
# scene_camera/left_wrist_camera/right_wrist_camera; openpi wants head/left/right.
_CAM_TO_OPENPI = {
    "scene_camera": "observation/image_head",
    "left_wrist_camera": "observation/image_left_wrist",
    "right_wrist_camera": "observation/image_right_wrist",
}

_DOF = 7  # per arm: 6 revolute + 1 gripper (matches raiden.server.DOF)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "t", "yes", "y", "on"):
            return True
        if value in ("0", "false", "f", "no", "n", "off"):
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


def _snapshot_obs_dict(obs_dict: dict) -> dict:
    """Copy the openpi observation before sending it from a worker thread."""

    out = {}
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray):
            out[key] = np.array(value, copy=True)
        else:
            out[key] = copy.deepcopy(value)
    return out


class OpenPIBridge(ModelBridge):
    """Drives a pi05 (openpi) policy server inside raiden's `rd infer` loop."""

    def __init__(self) -> None:
        self._ws = None
        self._prompt = ""
        self._horizon: int | None = None  # None => execute the full served chunk
        self._chunk: np.ndarray | None = None  # cached (chunk_len, 14)
        self._step = 0
        self._last_action: np.ndarray | None = None
        self._realtime_chunking = False
        self._replan_margin: int | None = None
        self._chunk_overrun_behavior = "hold"
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._next_future: concurrent.futures.Future | None = None
        self._ws_lock = threading.Lock()
        self._generation = 0

    # -- ModelBridge API ------------------------------------------------------

    def load(self, ckpt_path: str, **kwargs) -> None:
        """Connect to the policy server. There is no local checkpoint to load.

        ``ckpt_path`` is repurposed as the server address (``rd infer`` requires
        ``--ckpt-path``): ``"localhost:8000"``, ``"localhost"``, or ``""``.
        Precedence for every setting: ``--bridge-kwargs`` > ``OPENPI_*`` env > ckpt_path
        > default.
        """
        host, port = self._parse_addr(ckpt_path)
        host = str(kwargs.get("host") or os.environ.get("OPENPI_HOST") or host)
        port = int(kwargs.get("port") or os.environ.get("OPENPI_PORT") or port)

        self._prompt = str(kwargs.get("prompt", os.environ.get("OPENPI_PROMPT", "")))
        if not self._prompt:
            print(
                "[OpenPIBridge] WARNING: no prompt set — pi05 is language-conditioned, so "
                "an empty prompt will not reproduce training behavior. Pass "
                '--bridge-kwargs \'{"prompt": "..."}\' (keep it identical to '
                "--dagger-instruction)."
            )

        _ah = kwargs.get("action_horizon", os.environ.get("OPENPI_ACTION_HORIZON"))
        self._horizon = int(_ah) if _ah else None

        self._realtime_chunking = _as_bool(
            kwargs.get(
                "realtime_chunking",
                os.environ.get("OPENPI_REALTIME_CHUNKING", "false"),
            )
        )
        _margin = kwargs.get("replan_margin", os.environ.get("OPENPI_REPLAN_MARGIN"))
        self._replan_margin = int(_margin) if _margin else None
        self._chunk_overrun_behavior = str(
            kwargs.get(
                "chunk_overrun_behavior",
                os.environ.get("OPENPI_CHUNK_OVERRUN_BEHAVIOR", "hold"),
            )
        )
        if self._chunk_overrun_behavior not in ("hold", "block"):
            raise ValueError("chunk_overrun_behavior must be 'hold' or 'block'")
        if self._realtime_chunking and self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        from openpi_client import websocket_client_policy as _ws

        print(f"[OpenPIBridge] connecting to openpi server {host}:{port} ...")
        self._preflight(host, port)
        self._ws = _ws.WebsocketClientPolicy(host=host, port=port)
        print(
            f"[OpenPIBridge] connected | metadata={self._ws.get_server_metadata()} | "
            f"prompt={self._prompt!r} | action_horizon="
            f"{self._horizon if self._horizon else 'full served chunk'} | "
            f"realtime_chunking={self._realtime_chunking}"
            + (
                f" | replan_margin={self._replan_margin or 'half horizon'} | overrun={self._chunk_overrun_behavior}"
                if self._realtime_chunking
                else ""
            )
        )

    def predict(self, obs) -> np.ndarray:
        """One control step: obs -> (14,) absolute joint action [l7, r7]."""
        if self._realtime_chunking:
            return self._predict_realtime(obs)

        if self._chunk is None:
            self._install_chunk(self._query(obs))
        action = self._chunk[self._step].copy()
        self._step += 1
        if self._step >= (self._horizon or self._chunk.shape[0]):
            self._chunk = None  # exhausted -> next predict() re-queries the server
        return np.asarray(action, dtype=np.float32)

    def reset(self) -> None:
        """Handback / episode reset: drop the cached chunk so the next predict()
        re-infers on the CURRENT (post-correction) observation instead of replaying
        stale actions aimed at the pre-intervention pose."""
        self._generation += 1
        if self._next_future is not None:
            self._next_future.cancel()
        self._chunk, self._step = None, 0
        self._last_action = None
        self._next_future = None

    # -- helpers --------------------------------------------------------------

    def _predict_realtime(self, obs) -> np.ndarray:
        """Single control step with background prefetch of the next action chunk."""
        if self._chunk is None:
            installed = self._install_pending_if_ready(block=self._chunk_overrun_behavior == "block")
            if not installed:
                if self._last_action is not None and self._chunk_overrun_behavior == "hold":
                    return self._last_action.copy()
                self._install_chunk(self._query(obs))

        effective_horizon = self._effective_horizon()
        remaining = effective_horizon - self._step
        if self._next_future is None and remaining <= self._effective_replan_margin(effective_horizon):
            self._submit_next_chunk(obs)

        action = self._chunk[self._step].copy()
        self._last_action = action
        self._step += 1

        if self._step >= effective_horizon and not self._install_pending_if_ready(block=False):
            self._chunk = None

        return np.asarray(action, dtype=np.float32)

    def _install_chunk(self, chunk: np.ndarray) -> None:
        self._chunk = chunk
        self._step = 0

    def _effective_horizon(self) -> int:
        assert self._chunk is not None
        return self._horizon or self._chunk.shape[0]

    def _effective_replan_margin(self, horizon: int) -> int:
        if self._replan_margin is None:
            return max(1, horizon // 2)
        if self._replan_margin < 0 or self._replan_margin > horizon:
            raise ValueError("replan_margin must be between 0 and the effective action horizon")
        return self._replan_margin

    def _submit_next_chunk(self, obs) -> None:
        assert self._executor is not None
        generation = self._generation
        obs_dict = _snapshot_obs_dict(self._build_obs(obs))
        self._next_future = self._executor.submit(self._query_built_obs, obs_dict, generation)

    def _query_built_obs(self, obs_dict: dict, generation: int) -> tuple[int, np.ndarray]:
        return generation, self._query_obs_dict(obs_dict)

    def _install_pending_if_ready(self, *, block: bool) -> bool:
        if self._next_future is None:
            return False
        if not block and not self._next_future.done():
            return False

        future = self._next_future
        self._next_future = None
        generation, chunk = future.result()
        if generation != self._generation:
            return False
        self._install_chunk(chunk)
        return True

    @staticmethod
    def _parse_addr(spec: str) -> tuple[str, int]:
        """``"host:port"`` / ``"host"`` / ``""`` -> (host, port), defaulting to
        localhost:8000."""
        spec = (spec or "").strip().removeprefix("ws://").removeprefix("http://")
        if not spec:
            return "localhost", 8000
        if ":" in spec:
            host, _, port = spec.rpartition(":")
            return (host or "localhost"), int(port)
        return spec, 8000

    @staticmethod
    def _preflight(host: str, port: int, timeout_s: float = 3.0) -> None:
        """Fail fast if no server is listening.

        ``WebsocketClientPolicy._wait_for_server()`` retries FOREVER on a refused
        connection (``while True`` + 5 s sleep, logging at INFO — usually invisible), so
        a policy server that isn't running makes `rd infer` look like it hung with no
        output, before any hardware is brought up. One cheap TCP probe turns that into an
        actionable error.
        """
        import socket

        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return
        except OSError as exc:
            raise RuntimeError(
                f"no openpi policy server reachable at {host}:{port} ({exc}). "
                "Start it first, e.g.:\n"
                "  cd ~/robometer-policy-learning && uv run --no-sync python "
                "third_party/dsrl_openpi/scripts/serve_policy.py policy:checkpoint "
                "--policy.config=pi05_yam_pickbanana "
                "--policy.dir=~/pi05_yam_pickbanana/6000"
            ) from exc

    def _query(self, obs) -> np.ndarray:
        """Send one observation, return the validated (chunk_len, 14) action chunk."""
        return self._query_obs_dict(self._build_obs(obs))

    def _query_obs_dict(self, obs_dict: dict) -> np.ndarray:
        """Send one prepared observation, return the validated action chunk."""
        assert self._ws is not None, "load() must be called before predict()"
        with self._ws_lock:
            chunk = np.asarray(self._ws.infer(obs_dict)["actions"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != _DOF * 2:
            raise RuntimeError(f"expected a (chunk, {_DOF * 2}) action chunk from the openpi server, got {chunk.shape}")
        # The requested horizon must never exceed what the checkpoint emits: dispensing
        # chunk[i] for i >= chunk_len would IndexError mid-rollout with the arm driven.
        # pi05_yam_pickbanana serves 10 (Pi0Config action_horizon=10), so an inherited
        # larger value just means re-querying more often.
        if self._horizon and self._horizon > chunk.shape[0]:
            print(
                f"[OpenPIBridge] WARNING: action_horizon {self._horizon} exceeds the "
                f"checkpoint's chunk length {chunk.shape[0]} — clamping (server queried "
                "more often).",
                flush=True,
            )
            self._horizon = chunk.shape[0]
        return chunk

    def _build_obs(self, obs) -> dict:
        """raiden Observation -> openpi obs dict (3 letterboxed images + 14-D state)."""
        from openpi_client import image_tools

        # chiral's Observation.__getitem__ raises a bare KeyError('scene_camera') that
        # says nothing about WHY — and the usual cause is an unconfigured
        # ~/.config/raiden/camera.json (an empty `{}` yields zero cameras, so every
        # lookup fails). Name the gap and the fix instead.
        available = [c.name for c in obs.cameras]
        missing = [n for n in _CAM_TO_OPENPI if n not in available]
        if missing:
            raise RuntimeError(
                f"observation is missing camera(s) {missing}; it has {available or 'NONE'}. "
                "pi05 needs all three of "
                f"{sorted(_CAM_TO_OPENPI)}. Fix ~/.config/raiden/camera.json — the NAMES "
                "must match exactly (they key both the openpi observation mapping and the "
                "right_wrist 180° flip). Run `rd list_devices` for connected serials; see "
                "docs/guide/hardware.md for the format."
            )

        out = {}
        for cam_name, key in _CAM_TO_OPENPI.items():
            img = np.asarray(obs[cam_name].image)
            # Letterbox with the SAME function openpi applies internally, so a 224x224
            # input is passed through untouched server-side. No flip (already applied at
            # capture), no cv2.resize (would squash). See module docstring.
            out[key] = image_tools.resize_with_pad(img, MODEL_IMG_SIZE, MODEL_IMG_SIZE)
        out["observation/state"] = self._state(obs)
        if self._prompt:
            out["prompt"] = self._prompt
        return out

    @staticmethod
    def _state(obs) -> np.ndarray:
        """MEASURED joints -> openpi's (14,) [l6, l_grip, r6, r_grip].

        Measured (``_joint_pos``), not commanded (``_joint_cmd``): the LeRobot dataset's
        lowdim joints are measured, so the policy was conditioned on those.
        """
        left = np.asarray(obs.proprios["follower_l_joint_pos"], dtype=np.float32)
        right = np.asarray(obs.proprios["follower_r_joint_pos"], dtype=np.float32)
        state = np.concatenate([left, right])
        if state.shape != (_DOF * 2,):
            raise RuntimeError(
                f"expected a ({_DOF * 2},) bimanual state from "
                f"follower_l/r_joint_pos, got {state.shape} "
                f"(left={left.shape}, right={right.shape})"
            )
        return state
