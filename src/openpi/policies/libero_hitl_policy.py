"""LIBERO input/output transforms for human-in-the-loop (HITL) training of pi0 / pi0.5.

This is a copy of :mod:`openpi.policies.libero_policy` that additionally preserves a per-frame
``intervention`` label (0 = policy, 1 = human correction, 2 = offline demo) through the data
pipeline. The stock :class:`LiberoInputs` builds a fresh ``inputs`` dict containing only
state/image/actions/prompt, so any extra ``intervention`` key is dropped there; :class:`LiberoHitlInputs`
carries it forward so a HITL loss (e.g. Flow-MILE's intervention probit) can read it in the train step.

NOTE (Flow-MILE scaffold): passing ``intervention`` this far is necessary but NOT sufficient. The
label also has to survive the final data-loader hand-off, which today yields only
``(Observation, Actions)`` and drops everything else (``Observation.from_dict`` whitelists keys). See
the TODO anchors in ``scripts/train.py`` / ``src/openpi/models/model.py`` for the remaining wiring.
"""

import dataclasses

from openpi.models import model as _model
from openpi.policies.libero_policy import LiberoOutputs, _parse_image, make_libero_example  # noqa: F401
from openpi import transforms


@dataclasses.dataclass(frozen=True)
class LiberoHitlInputs(transforms.DataTransformFn):
    """Like :class:`openpi.policies.libero_policy.LiberoInputs`, but preserves ``intervention``.

    Identical behaviour to ``LiberoInputs`` for state/image/actions/prompt; the only addition is the
    ``intervention`` passthrough at the end. Kept as a separate class (rather than editing the stock
    one) so the native HG-DAgger path is untouched.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": _parse_image(data["observation/wrist_image"]) * 0,  # zero pad
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": self.model_type == _model.ModelType.PI0_FAST,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # HITL addition: carry the per-frame intervention label (0/1/2) forward for the HITL loss.
        if "intervention" in data:
            inputs["intervention"] = data["intervention"]

        return inputs
