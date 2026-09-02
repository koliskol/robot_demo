# Modality config for the Galbot G1 pickup_policy / place_policy datasets (see
# /home/kholis/Robot_project_streaming/CLAUDE.md's "LeRobot pick-and-place pipeline" for the
# source project). Both datasets share one embodiment (same robot, same 22-dim state/action
# layout - only the recorded task/behavior differs), so one shared NEW_EMBODIMENT registration
# covers both; point --dataset-path at whichever converted dataset you're fine-tuning on.
#
# Modeled on examples/SO100/so100_config.py - see that file and
# getting_started/finetune_new_embodiment.md for the general pattern.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


galbot_g1_config = {
    # Video: single head-mounted camera (see meta/modality.json's "video" entry).
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_camera"],
    ),
    # State: current proprioceptive reading. chassis_forward is the odd one out - it's the
    # chassis's cumulative signed forward-axis displacement (meters) since the current recording
    # attempt started, not a joint angle - see CLAUDE.md's chassis_forward section for the full
    # derivation (why it exists, why it's asymmetric with the action side below).
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "torso", "left_gripper", "right_gripper", "chassis_forward"],
    ),
    # Action: 16-step prediction horizon (matching the SO100 example's default - a starting
    # point, not verified against this dataset). One ActionConfig per modality key, same order as
    # modality_keys above.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm", "torso", "left_gripper", "right_gripper", "chassis_forward"],
        action_configs=[
            # left_arm/right_arm/torso: recorded as the post-safety-clamp TARGET joint position
            # sent to the controller each tick (not a delta from the previous position) - ABSOLUTE
            # is what we actually recorded, not a stylistic choice like SO100's RELATIVE pick.
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # grippers: target position, same reasoning as SO100's gripper entry.
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            # chassis_forward: doesn't cleanly fit this taxonomy at all - it's a recorded drive
            # *velocity* command (roughly [-drive_speed, +drive_speed]), not a position target in
            # any sense, relative or absolute. ABSOLUTE/NON_EEF is the closest approximation (take
            # the recorded number as-is, no EEF/rotation semantics apply) - unverified whether
            # GR00T's normalization/diffusion head handles a velocity-typed channel labeled
            # ABSOLUTE sensibly; worth watching this dimension specifically during open-loop eval.
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
        ],
    ),
    # Language: task instruction, sourced from the dataset's existing task_index column (see
    # meta/modality.json's "annotation" entry) - no extra annotation work needed, convert_to_lerobot.py
    # already writes one task per episode ("pickup_policy" or "place_policy").
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(galbot_g1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
