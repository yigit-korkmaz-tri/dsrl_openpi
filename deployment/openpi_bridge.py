"""OpenPI Pi0.5 bridge for YAM robot deployment.

Implements the ``ModelBridge`` interface from raiden. Connects to a running
OpenPI policy server (``serve_policy.py``) via websocket and translates
between raiden observations and the OpenPI observation format.

The OpenPI server handles all model inference, normalization, and
delta→absolute action conversion. This bridge is purely an observation
reformatter and action reorderer.

Usage (server + bridge)::

    # Terminal 1: start the OpenPI policy server
    python scripts/serve_policy.py \\
        --policy.config pi05_yam \\
        --policy.dir /path/to/checkpoint \\
        --port 8000

    # Terminal 2: run raiden inference
    rd infer \\
        --bridge deployment.openpi_bridge:OpenPiBridge \\
        --ckpt_path unused \\
        --action_hz 50.0 \\
        --host localhost --port 8000 \\
        --action_horizon 10 \\
        --prompt "pick up the lock and put it into the box"

OpenPI 14D layout: [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
Raiden 14D layout: [right_joint(6), right_grip(1), left_joint(6), left_grip(1)]
"""

from typing import Optional

import cv2
import numpy as np

from raiden.inference import ModelBridge

MODEL_IMG_SIZE = 224

# Camera name mapping: raiden camera name → OpenPI observation key.
_CAM_MAP: dict[str, str] = {
    "head": "observation/image_head",
    "left_wrist": "observation/image_left_wrist",
    "right_wrist": "observation/image_right_wrist",
}


def _raiden_to_openpi_state(
    r_joint_pos: np.ndarray,
    l_joint_pos: np.ndarray,
) -> np.ndarray:
    """Convert raiden proprio (7,) pairs to OpenPI 14D state.

    Raiden: r_joint_pos = [r_joints(6), r_grip(1)]
            l_joint_pos = [l_joints(6), l_grip(1)]
    OpenPI: [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
    """
    return np.concatenate([
        l_joint_pos[:6],
        l_joint_pos[6:7],
        r_joint_pos[:6],
        r_joint_pos[6:7],
    ]).astype(np.float32)


def _openpi_action_to_raiden(action_14d: np.ndarray) -> np.ndarray:
    """Convert OpenPI 14D action to raiden 14D motor command.

    OpenPI: [l_joints(6), l_grip(1), r_joints(6), r_grip(1)]
    Raiden: [r_joints(6), r_grip(1), l_joints(6), l_grip(1)]
    """
    return np.concatenate([
        action_14d[7:14],   # right joints (6) + right gripper (1)
        action_14d[0:7],    # left joints (6) + left gripper (1)
    ]).astype(np.float32)


class OpenPiBridge(ModelBridge):
    """ModelBridge implementation for OpenPI Pi0.5 on YAM.

    Connects to a running OpenPI websocket server and translates between
    raiden observations and OpenPI's expected input format.
    """

    def __init__(self, action_horizon: int = 10):
        self._broker = None
        self._action_horizon = action_horizon
        self._prompt: str = ""
        self._step = 0
        self._n_infer = 0
        self._t_infer_sum = 0.0

    def load(self, ckpt_path: str, **kwargs) -> None:
        """Connect to the OpenPI policy server.

        The ckpt_path is unused (the server loads the model). Connection
        parameters are passed via kwargs.

        Keyword args:
            host: Server host (default: "localhost").
            port: Server port (default: 8000).
            action_horizon: Number of actions to execute per inference call.
            prompt: Language instruction for the task.
        """
        from openpi_client import action_chunk_broker
        from openpi_client import websocket_client_policy as _ws

        host = kwargs.get("host", "localhost")
        port = int(kwargs.get("port", 8000))
        action_horizon = int(kwargs.get("action_horizon", self._action_horizon))
        self._prompt = kwargs.get("prompt", "")

        print(f"[openpi_bridge] Connecting to server at {host}:{port}")
        ws_policy = _ws.WebsocketClientPolicy(host=host, port=port)

        metadata = ws_policy.get_server_metadata()
        print(f"[openpi_bridge] Server metadata: {metadata}")

        self._broker = action_chunk_broker.ActionChunkBroker(
            policy=ws_policy,
            action_horizon=action_horizon,
        )
        print(f"[openpi_bridge] Ready (action_horizon={action_horizon})")

    def reset(self) -> None:
        """Reset action chunk buffer."""
        if self._broker is not None:
            self._broker.reset()
        self._step = 0
        self._n_infer = 0
        self._t_infer_sum = 0.0

    def predict(self, obs) -> np.ndarray:
        """Convert raiden observation → OpenPI input → inference → 14D action.

        Returns (14,) float32 motor command in raiden order:
        [r_joints(6), r_grip(1), l_joints(6), l_grip(1)]
        """
        import time

        t0 = time.perf_counter()
        obs_dict = self._preprocess(obs)
        result = self._broker.infer(obs_dict)
        infer_ms = (time.perf_counter() - t0) * 1e3

        self._t_infer_sum += infer_ms
        self._step += 1

        if self._step % 50 == 1:
            print(
                f"[openpi_bridge] step={self._step:4d}  "
                f"infer={infer_ms:.1f}ms  "
                f"avg={self._t_infer_sum / self._step:.1f}ms"
            )

        actions = result["actions"]  # (14,) from ActionChunkBroker
        return _openpi_action_to_raiden(actions)

    def _preprocess(self, obs) -> dict:
        """Convert raiden Observation to OpenPI observation dict."""
        obs_dict: dict = {}

        # Images: BGR uint8 → RGB uint8, resize to 224x224
        for cam in obs.cameras:
            obs_key = _CAM_MAP.get(cam.name)
            if obs_key is None:
                continue
            rgb = cv2.cvtColor(cam.image, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (MODEL_IMG_SIZE, MODEL_IMG_SIZE))
            obs_dict[obs_key] = rgb  # uint8 [224, 224, 3]

        # State: assemble 14D from raiden proprios
        r_pos = obs.proprios.get(
            "follower_r_joint_pos", np.zeros(7, dtype=np.float32)
        )
        l_pos = obs.proprios.get(
            "follower_l_joint_pos", np.zeros(7, dtype=np.float32)
        )
        obs_dict["observation/state"] = _raiden_to_openpi_state(r_pos, l_pos)

        # Language prompt
        if self._prompt:
            obs_dict["prompt"] = self._prompt

        return obs_dict


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy OpenPI Pi0.5 policy on YAM robot"
    )
    parser.add_argument("--ckpt_path", default="unused", help="Unused (server loads model)")
    parser.add_argument("--host", default="localhost", help="OpenPI server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPI server port")
    parser.add_argument("--action_horizon", type=int, default=10, help="Actions per inference")
    parser.add_argument("--action_hz", type=float, default=50.0)
    parser.add_argument("--prompt", default="", help="Language instruction")
    parser.add_argument("--camera_config_file", default="./config/camera_config.json")
    parser.add_argument("--calibration_file", default="./config/calibration_results.json")
    parser.add_argument("--stereo_method", default="zed", choices=["zed", "ffs"])
    parser.add_argument("--depth_mode", default="NEURAL_LIGHT")

    args = parser.parse_args()

    from raiden.inference import RaidenInferenceLoop

    bridge = OpenPiBridge(action_horizon=args.action_horizon)

    loop = RaidenInferenceLoop(
        bridge=bridge,
        ckpt_path=args.ckpt_path,
        action_hz=args.action_hz,
        bridge_kwargs={
            "host": args.host,
            "port": args.port,
            "action_horizon": args.action_horizon,
            "prompt": args.prompt,
        },
        camera_config_file=args.camera_config_file,
        calibration_file=args.calibration_file,
        stereo_method=args.stereo_method,
        depth_mode=args.depth_mode,
    )
    loop.run()


if __name__ == "__main__":
    main()
