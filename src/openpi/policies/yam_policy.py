"""Policy inputs/outputs for the YAM bimanual robot.

State/action: 14D joint positions [left_joint(6), left_grip(1), right_joint(6), right_grip(1)]
Cameras: head (base view), left_wrist, right_wrist
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yam_example() -> dict:
    """Creates a random input example for the YAM policy."""
    return {
        "observation/state": np.random.rand(14).astype(np.float32),
        "observation/image_head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the lock and put it into the box",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YAMInputs(transforms.DataTransformFn):
    """Inputs for the YAM bimanual policy.

    Expected inputs:
    - observation/image_head: base camera image
    - observation/image_left_wrist: left wrist camera image
    - observation/image_right_wrist: right wrist camera image
    - observation/state: [14] joint positions
    - actions: [action_horizon, 14] (only during training)
    - prompt: language instruction
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image_head"])
        left_wrist_image = _parse_image(data["observation/image_left_wrist"])
        right_wrist_image = _parse_image(data["observation/image_right_wrist"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YAMOutputs(transforms.DataTransformFn):
    """Outputs for the YAM bimanual policy.

    Returns the first 14 action dimensions (matching the 14D joint space).
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
 