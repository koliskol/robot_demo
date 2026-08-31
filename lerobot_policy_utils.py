"""Shared checkpoint-loading helper for anything that runs policy inference in the `lerobot` conda
env - used by both evaluate_act_checkpoint.py (offline open-loop replay) and policy_server.py
(closed-loop rollout inference server for collect_pickplace_demo.py, see CLAUDE.md's "LeRobot
pick-and-place pipeline"). Zero Isaac Sim imports, same as those two callers.
"""

from pathlib import Path

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors


def load_policy(checkpoint_dir: str, dataset_root: str, dataset_repo_id: str, device: str = "cuda") -> tuple:
    """Load a trained policy plus its pre/post-processor pipeline from a saved pretrained_model/
    directory (e.g. .../checkpoints/last/pretrained_model).

    dataset_root/dataset_repo_id must point at the (already-converted, see convert_to_lerobot.py)
    LeRobotDataset this checkpoint was trained on - make_policy() requires either dataset metadata
    or an env config to infer feature shapes, and this project has no env config (see
    evaluate_act_checkpoint.py's module docstring for why: no closed-loop env is registered for
    this custom Isaac Sim scene). The pre/post-processor's actual normalization stats come from the
    checkpoint itself (policy_preprocessor_step_3_normalizer_processor.safetensors etc.), not from
    ds_meta - ds_meta is only used for feature shape inference here.

    Returns (policy, preprocessor, postprocessor, camera_key) - camera_key is the dataset's full
    image feature name (e.g. "observation.images.head_camera"), ready to use as an observation dict
    key.
    """
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    policy_cfg.pretrained_path = checkpoint_dir
    policy_cfg.device = device
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    policy = make_policy(policy_cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=checkpoint_dir)
    camera_key = ds_meta.camera_keys[0]
    return policy, preprocessor, postprocessor, camera_key
