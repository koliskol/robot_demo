# Tutorial: collecting pick-and-place data for a LeRobot policy

This walks through using `collect_pickplace_demo.py`, `check_raw_episodes.py`, and
`convert_to_lerobot.py` to record teleoperated demonstrations of a Galbot G1 robot picking a box
up off a table, carrying it (bimanual hug, not the gripper) to a pushcart, and vice versa — and
turning those recordings into a dataset a LeRobot policy (e.g. ACT) can train on.

If you haven't read it yet, `CLAUDE.md`'s "LeRobot pick-and-place pipeline" section covers the
*why* behind the design choices here (why a hug instead of a gripper pinch, why the cart height is
a tunable, why conversion runs in a separate environment). This document is the *how*.

## Prerequisites

- An `isaac_sim` conda environment with Isaac Sim installed (used for `collect_pickplace_demo.py`
  only — no `lerobot` needed here). It also needs `aiortc`/`aiohttp`/`av` for the optional WebRTC
  camera viewer below (same deps `stream_demo.py` needs) — these should already be present if
  `stream_demo.py` has ever worked in this env.
- Network access for Isaac Sim to resolve its Nucleus assets root path.
- Later, for conversion/training: a separate environment with `pip install lerobot` — see
  Step 4.

## Watching the robot's camera feed while collecting

`collect_pickplace_demo.py` starts the same WebRTC server `stream_demo.py` uses
(`streaming_server.py`), but serves its own dedicated page — just the RGB feed, no depth/lidar/map
panels at all (this task doesn't use those sensors, and unlike `stream_demo.py`'s page, they're
not just hidden here, they don't exist on the page). Open
`http://<this machine's address>:8080/` (or whatever `--host`/`--port` you passed) and click
Connect to watch live while teleoperating.

Below the video, a status bar appears (idle / recording / awaiting label, with episode number and
frame count) along with **Start Recording / Success (Y) / Fail (F) / Discard** buttons — these
drive the exact same recorder as the `B`/`Y`/`F`/`Backspace` keys, so you can control recording
from the browser instead of (or alongside) the keyboard. Watching the stream itself is still
purely for convenience either way — it has no effect on what actually gets recorded (the recorder
samples the same camera independently, at a fixed rate, regardless of whether anyone's watching).

## Step 0 — smoke test before recording anything real

Launch the scene with no recording yet:

```
conda run -n isaac_sim python collect_pickplace_demo.py --cube-start table
```

This opens the Isaac Sim viewport. The console prints a one-time geometry diagnostic:

```
[geometry] table_top_z=0.720m  cart_deck_top_z=0.150m  delta=+0.570m  cube_scale=1.0 ...
```

With the viewport focused, manually jog through **one full cycle** — don't worry about recording:

1. `I`/`K` — crouch/raise the torso. `U`/`O` — swing both arms forward/back (this is the hug
   motion). `J`/`L` — raise/lower both hands (elbow only).
2. Bring both arms forward around the box on the table (`U`) until it's compressed between the
   forearms. This friction hold is the primary grip — `M`/`N` (gripper close/open) is optional
   extra contact, not required.
3. Carry it toward the pushcart (torso/arm adjustments as needed) and release (`O` to swing back
   open).
4. Try the reverse: relaunch with `--cube-start cart` and hug-carry it back to the table.

Things to check, and what to do if they fail:

| Problem | Fix |
|---|---|
| Box not visible in the viewport | Select `/World/Cube` in the Stage panel, press `F` to frame it. It's a real warehouse cardboard-box asset now, not a tiny procedural cube - `--cube-scale` resizes it (default 1.0 = native size, ~0.38 x 0.25 x 0.15m). |
| Arms can't reach low/close enough to grip on the table, or can't clear the cart deck | Relaunch with a different `--deck-riser <meters>` (raises the cart deck without touching its wheels). |
| The two arms don't converge enough around the box to hold it | This needs adjusting `ARM_FORWARD_POSE`'s shoulder angle in the script — a code change, not a CLI flag. Note how far off it is and we can tune it. |
| Robot body collides with table/cart, or can't reach at all | Adjust `ROBOT_APPROACH_GAP_M`/`CART_TABLE_GAP_M` constants near the top of the script. |
| Hold is unstable / box flies out of the hug | Lower `--cube-mass`, or bind a custom high-friction `PhysicsMaterial` onto the box (not done by default - see `spawn_real_box`'s docstring), before trying anything more exotic. Do **not** add a joint-based attach — see `CLAUDE.md`. |
| Box/hand invisible once very close to the camera | Was Isaac Sim's default 1.0m near clipping plane - anything closer than 1m to the lens wasn't rendered at all. Fixed via `camera.set_clipping_range()` (`CAMERA_NEAR_CLIP_M = 0.1`) - confirmed live (an object ~0.4m away rendered as nothing before, visible after). Don't set this lower without testing rendered brightness first - 0.02 (and 0.03, 0.05) were tried and made the whole frame go almost black with no error, see `CLAUDE.md`. |
| A dark curved shape intrudes into the frame even at rest pose (no crouch, no arm swing) | That's the camera seeing part of its own head housing, not a bug - root-caused via a physics raycast through the exact screen region, which hit `head_link2`'s own collision mesh on every ray. Fixed by nudging the camera forward along the mount's local +X (`CAMERA_MOUNT_FORWARD_OFFSET_M = 0.1`) - confirmed both directions live: 0.0 shows the obstruction, +0.1 fully clears it, -0.1 makes it fill most of the frame instead. Nothing to change here, this is already applied. |
| A dark shape fills most of the frame during a deep crouch + arm swing | A separate issue from the one above: that's the robot's own torso/shoulder self-occluding the head-mounted camera, not a bug - confirmed live via a controlled pose test. Tilting down further makes it worse, not better. Try less torso crouch (partial `I`/`K`), relying more on elbow lift (`J`/`L`) to keep the torso out of the camera's view - a teleop-technique fix, not something tunable via constants. |
| Camera view looks tilted sideways, not just down | Not the mount's roll/tilt math (verified correct via static rendering) - the head's own pan/tilt joints ship with very weak drive and, until fixed, nothing commanded them. During active torso/arm motion the head could wobble up to ~36° in an uncontrolled direction. Now stiffened and actively held every frame (`stiffen_head_joints`/`hold_head_joints`), cutting worst-case drift to ~12° under an aggressive stress test - real teleop should see less, but some residual wobble during fast motion is still expected. See `CLAUDE.md`. |

The boxes are real cardboard-box props from Isaac's warehouse/logistics asset set (generic
shipping boxes, not branded items) — plain colored cubes were dropped entirely. These ship as
static (collision-only) meshes, so `make_box_dynamic()` explicitly authors rigid-body physics and
overrides their mesh collision to `convexHull` (verified live: settles cleanly under gravity, no
instability). A decorative second table sits far across the room — it's not part of the task,
ignore it during this check. On `--cube-start table` sessions, two extra bigger boxes (distinct
real assets, `--cube2-scale`/`--cube3-scale` to resize further, spaced apart via `CUBE_ROW_GAP_M`)
also spawn alongside the main one, for size variety — these are real pick-up targets too, not
decoration; pass `--no-extra-boxes` for just the single box. They don't spawn on `--cube-start
cart` sessions (the pushcart deck is too small to fit 3 boxes) — the main box alone is small
enough to fit the deck (confirmed: 0.38m fits within the deck's 0.6m width).

Do not move on to real recording until one full pick → place → pick → place cycle works reliably.

## Step 1 — record episodes

Once Step 0 works, start a real session:

```
conda run -n isaac_sim python collect_pickplace_demo.py --cube-start table --deck-riser <tuned value> --out ./raw_episodes
```

Controls during recording:

| Key | Action |
|---|---|
| `W`/`S`/`A`/`D`/`Q`/`E` | drive/strafe/rotate — only for parking between episodes, not part of a recorded episode |
| `I`/`K` | torso up/down |
| `U`/`O` | swing both arms forward/back (the hug) |
| `J`/`L` | raise/lower both hands |
| `M`/`N` | close/open both grippers (optional) |
| `B` | toggle: start recording → stop and await a label |
| `Y` | (after stop) label the episode **success** and save it |
| `F` | (after stop) label the episode **failure** and save it |
| `Backspace` | (after stop) discard the episode, don't save |
| `R` | reset robot/cube/cart to spawn (also discards an in-progress episode) |

Per episode: jog into position → `B` (start) → perform the hug-carry → `B` (stop) →
`Y`/`F`/`Backspace` → `R` → repeat.

Episodes land in `./raw_episodes/episode_0000/`, `episode_0001/`, ... regardless of session, each
containing:

```
episode_0000/
  manifest.json           # success flag, fps, task name, joint names, frame count
  data.npz                # observation.state (T,21) and action (T,21) float32 arrays
  frames/NNNNNN_rgb.png   # RGB frame, one per recorded timestep
  frames/NNNNNN_depth.npy # raw depth (meters, float32, 480x640) for the same timestep -
                          # captured "just in case" a future policy wants it; not currently used
                          # by convert_to_lerobot.py (see that script's own docstring)
```

Depth also shows live in the browser viewer (a false-colored preview, same as `stream_demo.py`'s)
alongside RGB, purely for your own viewing convenience — the recorded depth is the raw float
array, not this colorized preview.

Then record the reverse direction in a second session:

```
conda run -n isaac_sim python collect_pickplace_demo.py --cube-start cart --out ./raw_episodes
```

Both sessions share `--out`, so episode numbering continues automatically — no manual bookkeeping
needed. Aim for roughly **20-30+ successful episodes per direction** as a starting point; more,
and more varied approach angles, generally helps.

## Step 2 — sanity-check what you recorded

No conda env required — plain `python3` (numpy + Pillow, already installed):

```
python3 check_raw_episodes.py --raw-dir ./raw_episodes
```

Run this after your first handful of episodes, before investing in a long session. It flags:
shape mismatches between the manifest and the actual data, NaN/Inf values, episodes where nothing
moved (state/action frozen — usually means recording started before any motion, or a key wasn't
held), and blank/near-uniform camera frames. Exits non-zero if anything's wrong, and prints every
problem found, not just the first.

## Step 3 — convert to a LeRobot dataset

This runs in a **separate environment**, not `isaac_sim` — see `CLAUDE.md` for why (`lerobot`'s
dependencies could conflict with Isaac Sim's pinned versions of numpy/torch/opencv/etc.):

```
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install lerobot
```

Before trusting the conversion script's exact API calls, check them against what actually got
installed (the LeRobotDataset API has changed across releases and wasn't verified live while this
script was written):

```
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; help(LeRobotDataset.create)"
```

Then convert:

```
python convert_to_lerobot.py --raw-dir ./raw_episodes --repo-id local/pick_box_table_to_cart --root ./lerobot_dataset
```

By default only episodes labeled **success** are included (`--include-failures` to add failures
too — they stay on disk either way, nothing is deleted).

## Step 4 — train

Still in the `lerobot` environment, once converted:

```
python -m lerobot.scripts.train --policy.type=act --dataset.repo_id=local/pick_box_table_to_cart \
  --output_dir=./outputs/act_pickplace --batch_size=8 --steps=100000
```

Check `lerobot-train --help` / `python -m lerobot.scripts.train --help` for the exact flags on
your installed version — this has also shifted across releases.

Close Isaac Sim before training — data collection and training are separate processes run at
different times, and the GPU may not have headroom for both at once (this machine has a 12GB
card, often already partly used by other work).

## Quick reference: file map

| File | Runs in | Purpose |
|---|---|---|
| `collect_pickplace_demo.py` | `isaac_sim` conda env | Scene + keyboard teleop + episode recorder |
| `check_raw_episodes.py` | plain `python3` | Validates recorded episodes before conversion |
| `convert_to_lerobot.py` | separate `lerobot` env | Raw episodes → LeRobotDataset |
