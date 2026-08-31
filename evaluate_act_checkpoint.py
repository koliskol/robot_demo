"""Offline sanity check for a trained LeRobot policy checkpoint (e.g. from train_act, see
CLAUDE.md's "LeRobot pick-and-place pipeline"): replays a handful of recorded episodes from
raw_episodes/ frame-by-frame through the policy's real inference path (policy.select_action(),
the same action-chunk queue behavior used in real deployment - see ACTConfig.chunk_size/
n_action_steps) and reports how closely the predicted actions track the recorded ones.

IMPORTANT: this is NOT a closed-loop rollout. The policy is fed the actual recorded observation
at every step, never what it would have seen after acting on its own (possibly imperfect) earlier
predictions - so a low error here does not by itself prove the policy can control the robot
end-to-end (errors compound differently in closed loop, and this never tests whether the hug/
grasp itself would actually succeed). It's a fast, no-Isaac-Sim-needed pre-check: if a policy
can't even track its own training data's actions from the *true* observations, it's not worth the
much larger effort of wiring up a real closed-loop rollout in Isaac Sim. Zero Isaac Sim imports on
purpose - run this in the `lerobot` conda env (same one used for convert_to_lerobot.py/training),
not isaac_sim.

    conda run -n lerobot python evaluate_act_checkpoint.py \
        --checkpoint-dir ./act_training/table_to_table2/checkpoints/last/pretrained_model \
        --raw-dir ./raw_episodes --dataset-root ./lerobot_dataset --dataset-repo-id local/pick_box_table_to_table2

By default, evaluates the first/middle/last successful episode found in --raw-dir. Pass
--episodes explicitly to pick specific ones. Note that unless you've held out episodes from
training, this measures in-sample tracking accuracy, not generalization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.utils.control_utils import predict_action

from lerobot_policy_utils import load_policy


def load_episode(episode_dir: Path) -> dict:
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    data = np.load(episode_dir / "data.npz")
    frame_paths = sorted((episode_dir / "frames").glob("*.png"))
    if len(frame_paths) != manifest["num_frames"]:
        raise ValueError(f"{episode_dir}: manifest says {manifest['num_frames']} frames but found {len(frame_paths)} PNGs")
    return {"manifest": manifest, "state": data["observation.state"], "action": data["action"], "frame_paths": frame_paths}


def pick_default_episodes(raw_dir: Path, n: int = 3) -> list:
    candidates = []
    for ep_dir in sorted(raw_dir.glob("episode_*")):
        manifest = json.loads((ep_dir / "manifest.json").read_text())
        if manifest["success"]:
            candidates.append(ep_dir.name)
    if not candidates:
        raise SystemExit(f"No success-labeled episodes found under {raw_dir}")
    if len(candidates) <= n:
        return candidates
    idxs = np.linspace(0, len(candidates) - 1, n).round().astype(int)
    return [candidates[i] for i in sorted(set(idxs))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to a saved pretrained_model/ directory (e.g. .../checkpoints/last/pretrained_model).")
    parser.add_argument("--raw-dir", type=str, default="./raw_episodes", help="Directory of episode_NNNN/ dirs to replay (same format check_raw_episodes.py reads).")
    parser.add_argument("--dataset-root", type=str, required=True, help="Root of the LeRobotDataset this checkpoint was trained on (needed to build the policy's feature spec).")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="repo_id of that same dataset.")
    parser.add_argument("--episodes", type=str, nargs="*", default=None, help="Specific episode_NNNN names to replay (default: first/middle/last success-labeled episode in --raw-dir).")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device.")
    args = parser.parse_args()

    device = torch.device(args.device)
    raw_dir = Path(args.raw_dir)
    episode_names = args.episodes if args.episodes else pick_default_episodes(raw_dir)
    print(f"Evaluating episodes: {episode_names}")

    policy, preprocessor, postprocessor, camera_key = load_policy(
        args.checkpoint_dir, args.dataset_root, args.dataset_repo_id, device=args.device
    )
    overall_errors = []

    for name in episode_names:
        ep = load_episode(raw_dir / name)
        manifest = ep["manifest"]
        state_names = manifest["state_names"]
        task = manifest["task"]
        policy.reset()

        errors = np.zeros((manifest["num_frames"], len(state_names)), dtype=np.float64)
        for t, frame_path in enumerate(ep["frame_paths"]):
            rgb = np.array(Image.open(frame_path).convert("RGB"))
            observation = {
                camera_key: rgb,
                "observation.state": ep["state"][t].astype(np.float32),
            }
            action = predict_action(
                observation, policy, device, preprocessor, postprocessor, use_amp=False, task=task,
            )
            predicted = action.cpu().numpy()
            errors[t] = predicted - ep["action"][t]

        abs_err = np.abs(errors)
        mae_per_dim = abs_err.mean(axis=0)
        overall_mae = abs_err.mean()
        overall_errors.append(overall_mae)
        print(f"\n[{name}] task={task!r} frames={manifest['num_frames']} overall MAE={overall_mae:.4f} rad")
        worst = np.argsort(-mae_per_dim)[:5]
        for i in worst:
            print(f"    {state_names[i]:>20s}: MAE={mae_per_dim[i]:.4f} rad")

    print(f"\nMean overall MAE across {len(episode_names)} episodes: {np.mean(overall_errors):.4f} rad")


if __name__ == "__main__":
    main()
