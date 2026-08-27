"""Stage 2 sanity check for episodes recorded by collect_pickplace_demo.py - run this on a small
test batch (2-3 episodes) before investing in a full collection session, and again on the full
set before converting with convert_to_lerobot.py.

Plain Python (numpy + Pillow only, no lerobot, no Isaac Sim) - runs fine outside any conda env:

    python3 check_raw_episodes.py --raw-dir ./raw_episodes

Checks, per episode:
  - manifest.json / data.npz / frames/*.png are internally consistent (frame counts match).
  - observation.state and action arrays have the expected shape and contain no NaN/Inf.
  - state and action aren't pinned at the same value every frame (a sign nothing actually moved,
    e.g. a key wasn't held long enough or the recorder started before any motion began).
  - RGB frames aren't blank/near-uniform (a sign the camera wasn't ready or nothing is in view).
  - If the manifest says depth was captured: frames/*_depth.npy count matches, and a sample of
    them contain no NaN (Inf is a legitimate "no-hit" depth reading, not flagged as a problem).

Exits non-zero if any episode fails a check, after printing every problem found (not just the
first) so you can fix a whole batch in one pass rather than one failure at a time.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def check_episode(episode_dir: Path) -> list:
    problems = []

    manifest_path = episode_dir / "manifest.json"
    data_path = episode_dir / "data.npz"
    frames_dir = episode_dir / "frames"
    if not manifest_path.exists():
        return [f"missing manifest.json"]
    if not data_path.exists():
        return [f"missing data.npz"]

    manifest = json.loads(manifest_path.read_text())
    num_frames = manifest["num_frames"]
    state_dim = manifest["state_dim"]
    action_dim = manifest["action_dim"]
    fps = manifest["fps"]

    if num_frames == 0:
        return ["manifest reports 0 frames"]

    data = np.load(data_path)
    for key, expected_dim in (("observation.state", state_dim), ("action", action_dim)):
        if key not in data:
            problems.append(f"data.npz missing '{key}'")
            continue
        arr = data[key]
        if arr.shape != (num_frames, expected_dim):
            problems.append(f"'{key}' shape {arr.shape} != expected {(num_frames, expected_dim)}")
            continue
        if not np.isfinite(arr).all():
            problems.append(f"'{key}' contains NaN/Inf values")
        if num_frames > 1 and np.allclose(arr, arr[0], atol=1e-4):
            problems.append(f"'{key}' is identical across all {num_frames} frames - nothing moved during this episode")

    frame_paths = sorted(frames_dir.glob("*.png")) if frames_dir.exists() else []
    if len(frame_paths) != num_frames:
        problems.append(f"found {len(frame_paths)} PNG frames but manifest says {num_frames}")

    # Spot-check a handful of frames (first, middle, last) rather than every one, for speed.
    sample_indices = sorted(set([0, len(frame_paths) // 2, len(frame_paths) - 1]) & set(range(len(frame_paths))))
    for i in sample_indices:
        img = np.array(Image.open(frame_paths[i]).convert("RGB"))
        if img.std() < 1.0:
            problems.append(f"frame {i} ({frame_paths[i].name}) looks blank/near-uniform (std={img.std():.3f})")

    if manifest.get("depth_capture", False):
        depth_paths = sorted(frames_dir.glob("*_depth.npy")) if frames_dir.exists() else []
        if len(depth_paths) != num_frames:
            problems.append(f"found {len(depth_paths)} depth .npy files but manifest says {num_frames}")
        for i in sample_indices:
            if i < len(depth_paths):
                depth = np.load(depth_paths[i])
                # Inf is a legitimate "no-hit" reading (see EpisodeRecorder.save) - only NaN
                # indicates an actual problem.
                if np.isnan(depth).any():
                    problems.append(f"depth frame {i} ({depth_paths[i].name}) contains NaN values")

    duration = num_frames / fps
    print(
        f"{episode_dir.name}: {'success' if manifest['success'] else 'FAILURE'}  "
        f"{num_frames} frames  {duration:.1f}s  task={manifest['task']!r}"
        + ("  -- OK" if not problems else "")
    )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=str, default="./raw_episodes", help="Directory of episode_NNNN/ dirs from collect_pickplace_demo.py.")
    args = parser.parse_args()

    episode_dirs = sorted(Path(args.raw_dir).glob("episode_*"))
    if not episode_dirs:
        raise SystemExit(f"No episode_* directories found under {args.raw_dir}")

    total_problems = 0
    total_frames = 0
    success_count = 0
    for ep_dir in episode_dirs:
        problems = check_episode(ep_dir)
        if problems:
            total_problems += len(problems)
            for p in problems:
                print(f"  PROBLEM: {p}")
        else:
            manifest = json.loads((ep_dir / "manifest.json").read_text())
            total_frames += manifest["num_frames"]
            success_count += int(manifest["success"])

    print()
    print(f"{len(episode_dirs)} episodes checked, {success_count} labeled success, {total_frames} total good frames.")
    if total_problems:
        print(f"{total_problems} problem(s) found - see above.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
