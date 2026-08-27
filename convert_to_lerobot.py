"""Convert raw episodes recorded by collect_pickplace_demo.py into a LeRobotDataset, for
training a LeRobot policy (e.g. ACT) on the table<->pushcart pick-and-place task.

Deliberately has ZERO Isaac Sim imports (no isaacsim/carb/omni/pxr) - run this in a separate,
lightweight environment where `pip install lerobot` is safe, not the isaac_sim conda env (which
has its own pinned numpy/torch/opencv/etc. that lerobot's dependencies could conflict with):

    conda create -n lerobot python=3.10 -y
    conda activate lerobot
    pip install lerobot
    python convert_to_lerobot.py --raw-dir ./raw_episodes --repo-id local/pick_box_table_to_cart --root ./lerobot_dataset

IMPORTANT: the exact LeRobotDataset API (module path, `create()`'s keyword arguments, whether
`task` is passed to add_frame or elsewhere, whether a finalize/consolidate call is needed) has
changed across lerobot releases and could not be verified against a real install while writing
this script (lerobot isn't installed anywhere on this machine yet). Before trusting this file,
run, in your lerobot env:

    python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; help(LeRobotDataset.create)"

(or the lerobot.common.datasets... path - this script tries both, see the import block below)
and adjust build_features()/main() below to match whatever you actually see. Treat everything
past the import block as a best-effort starting point, not a verified-working script.

NOT YET WIRED IN: collect_pickplace_demo.py records depth too (raw float32 meters,
frames/NNNNNN_depth.npy per frame, "just in case" a future policy wants it) but this converter
currently only builds RGB + state/action features - depth isn't added to the dataset. LeRobot
gained depth *dataset* support in v0.6.0 (RealSense-oriented, 12-bit depth video streams,
`use_depth: true`), but that's storage infrastructure, not confirmed support in the standard
policies (ACT/Diffusion/SmolVLA/etc. don't appear to consume depth as a model input out of the
box as of this writing) - so adding it here would mean guessing at an API this project has no way
to verify without a live lerobot install, for a policy that likely wouldn't use it anyway. If you
want depth in the trained dataset: check the installed lerobot version's actual depth feature API
first (same "don't trust this file" caveat as the RGB path above, doubly so here), then extend
load_raw_episode/build_features/main to read the `*_depth.npy` files already sitting on disk.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_raw_episode(episode_dir: Path) -> dict:
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    data = np.load(episode_dir / "data.npz")
    frame_paths = sorted((episode_dir / "frames").glob("*.png"))
    if len(frame_paths) != manifest["num_frames"]:
        raise ValueError(f"{episode_dir}: manifest says {manifest['num_frames']} frames but found {len(frame_paths)} PNGs")
    return {
        "manifest": manifest,
        "state": data["observation.state"],
        "action": data["action"],
        "frame_paths": frame_paths,
    }


def build_features(state_names: list, camera_key: str, height: int, width: int) -> dict:
    return {
        f"observation.images.{camera_key}": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, default="./raw_episodes", help="Directory of episode_NNNN/ dirs from collect_pickplace_demo.py.")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id (e.g. local/pick_box_table_to_cart).")
    parser.add_argument("--root", type=str, required=True, help="Local directory to write the converted LeRobotDataset into.")
    parser.add_argument("--success-only", action="store_true", default=True, help="Only convert episodes labeled success (default: on).")
    parser.add_argument("--include-failures", dest="success_only", action="store_false", help="Also convert episodes labeled failure.")
    args = parser.parse_args()

    episode_dirs = sorted(Path(args.raw_dir).glob("episode_*"))
    if not episode_dirs:
        raise SystemExit(f"No episode_* directories found under {args.raw_dir}")

    first = load_raw_episode(episode_dirs[0])
    manifest = first["manifest"]
    state_names = manifest["state_names"]
    camera_key = manifest["camera"]["key"]
    height, width = manifest["camera"]["height"], manifest["camera"]["width"]
    fps = manifest["fps"]

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=args.root,
        features=build_features(state_names, camera_key, height, width),
    )

    converted, skipped = 0, 0
    for ep_dir in episode_dirs:
        raw = load_raw_episode(ep_dir)
        if args.success_only and not raw["manifest"]["success"]:
            skipped += 1
            continue
        for t, frame_path in enumerate(raw["frame_paths"]):
            rgb = np.array(Image.open(frame_path).convert("RGB"))
            dataset.add_frame(
                {
                    f"observation.images.{camera_key}": rgb,
                    "observation.state": raw["state"][t],
                    "action": raw["action"][t],
                    "task": raw["manifest"]["task"],
                }
            )
        dataset.save_episode()
        converted += 1
        print(f"Converted {ep_dir.name} ({raw['manifest']['num_frames']} frames)")

    print(f"Done: {converted} episodes converted, {skipped} skipped (failures) -> {args.root}")


if __name__ == "__main__":
    main()
