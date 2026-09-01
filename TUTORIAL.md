# Tutorial: collecting pick-and-place data for a LeRobot policy

This walks through using `collect_pickplace_demo.py`, `check_raw_episodes.py`,
`convert_to_lerobot.py`, `evaluate_act_checkpoint.py`, and `policy_server.py` to record
teleoperated demonstrations of a Galbot G1 robot picking a box up off a table and placing it on a
pushcart or a second table (bimanual hug, not the gripper) — training a LeRobot policy (e.g. ACT)
on those recordings, and testing it both offline and live in Isaac Sim.

If you haven't read it yet, `CLAUDE.md`'s "LeRobot pick-and-place pipeline" section covers the
*why* behind the design choices here (why a hug instead of a gripper pinch, why conversion/
training run in a separate environment, why the recorded action space grew from 21 to 22 dims,
why pickup and place are now two separate policies). This document is the *how*.

## Prerequisites

- An `isaac_sim` conda environment with Isaac Sim installed (used for `collect_pickplace_demo.py`
  only — no `lerobot`/`torch` needed here, and installing it risks conflicting with Isaac Sim's own
  pinned deps). It also needs `aiortc`/`aiohttp`/`av` for the WebRTC camera viewer (same deps
  `stream_demo.py` needs) — these should already be present if `stream_demo.py` has ever worked in
  this env.
- Network access for Isaac Sim to resolve its Nucleus assets root path.
- For conversion/training/evaluation/rollout inference: a separate `lerobot` environment — see
  Step 3.

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
[geometry] table_top_z=0.517m  cart_deck_top_z=0.150m  delta=+0.367m  cube_scale=1.0 ...
```

`table_top_z` is lower than the table asset's native ~0.75m because of `--table-height-scale`
(default 0.69 - see the flag's own help text) - a height-only scale on the table's legs, added
for easier arm reach, that leaves the tabletop's footprint (0.8m x 2.8m) untouched. Pass
`--table-height-scale 1.0` for the native height, or any other factor to retune it.

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
| Only one hand is visible mid-hug even though the robot is facing straight | Not a camera bug - confirmed live with a symmetric wide-FOV render at rest (no yaw bias) and a controlled test that reproduced a fully-converged, hands-raised hug: both grippers land centered and symmetric in frame together. The single-hand view happens when the swing (`U`) hasn't fully closed yet and/or the hands haven't been raised (`J`) - a controlled test found the two arms don't enter the camera's view at the same rate while still mid-swing (their `ARM_FORWARD_POSE` joint values aren't simple mirrors of each other), so whichever arm is further along shows up first. Hold `U` and `J` together and give it a moment to settle before judging the framing. |

The boxes are real cardboard-box props from Isaac's warehouse/logistics asset set (generic
shipping boxes, not branded items) — plain colored cubes were dropped entirely. These ship as
static (collision-only) meshes, so `make_box_dynamic()` explicitly authors rigid-body physics and
overrides their mesh collision to `convexHull` (verified live: settles cleanly under gravity, no
instability). On `--cube-start table` sessions, two extra bigger boxes (distinct
real assets, `--cube2-scale`/`--cube3-scale` to resize further, spaced apart via `CUBE_ROW_GAP_M`)
also spawn alongside the main one, for size variety — these are real pick-up targets too, not
decoration; pass `--no-extra-boxes` for just the single box. They don't spawn on `--cube-start
cart` or `--cube-start table2` sessions (the pushcart deck is too small to fit 3 boxes, and the
extra boxes are only ever placed on table1's own surface) — the main box alone is small enough to
fit the pushcart deck (confirmed: 0.38m fits within the deck's 0.6m width).

The box's spawn/reset position and yaw are randomized a little each episode by default
(`--box-jitter-m`, default 0.03m radius; `--box-yaw-jitter-deg`, default 10°) — every episode
(including across `R`-triggered resets within a session) used to spawn the box at the exact same
point, which risks a policy that only ever learned one pixel-perfect box pose. This is on by
default; pass `0` to either flag to disable it. Console prints the sampled offset each episode.
Keep this modest — the hug's arm pose (`ARM_FORWARD_POSE`) is tuned for one specific position, so
if the hug stops reliably converging after widening these, narrow them back down.

### Pick/place target: pushcart or a second table

By default the box's destination is the pushcart. Pass `--place-target table2` instead to use a
second table (same asset/height as the main one) placed to table1's **side** — offset along
table1's short (0.8m) axis, not ahead along its long (2.8m) axis, so the robot approaches table2's
long 2.8m edge rather than its narrow end. The gap (`TABLE2_GAP_M`, 0.6m) is deliberately real now,
not a token clearance — since `chassis_forward` records the drive there, covering that distance is
part of what `place_policy` is supposed to learn, not something to design around. `TABLE2_SIDE_SIGN`
picks which side table2 sits on; flip it in the script if the layout looks backward once viewed
live. Only one of {cart, table2} is ever built per session — pick one and pass a matching
`--cube-start`:

```
conda run -n isaac_sim python collect_pickplace_demo.py --place-target table2 --cube-start table
conda run -n isaac_sim python collect_pickplace_demo.py --place-target table2 --cube-start table2
```

`--cube-start` must be `table` or match `--place-target` (e.g. `--cube-start cart` requires
`--place-target cart`, the default) — passing a mismatched pair exits with an argument error
rather than building an inconsistent scene. Since table2 is the same full-size table as table1
(0.8m × 2.8m — far too long to reach across from one parked pose), the actual pick/place point
sits `TABLE2_EDGE_INSET_M` onto its surface from the near edge, not its centroid; most of table2's
surface is just for visual realism, out of reach, same as a real table is bigger than its contact
patch.

Do not move on to real recording until one full pick → place → pick → place cycle works reliably.

## Two policies, not one continuous trajectory: pickup and place

The task is split into **two independent, fixed-base policies**, handed off between by a
navigation/SLAM stack that isn't part of this repo:

- **`pickup`**: open arm → approach the box → hug → lift slightly → back away → end (SLAM takes
  over from here).
- **`place`**: SLAM has just parked the robot near the destination, robot is already holding the
  box → approach the table/cart → lower slightly → open the arm wide (release) → back away → end
  (SLAM takes over again).

This means recording sessions for the two are structured differently, and **both now include the
robot's own forward/backward chassis motion as part of what's recorded** (see the next section) —
approach and retreat are supposed to be driven by the policy, not left to SLAM, so `W`/`S` during a
recording is no longer just "parking between episodes," it's part of the demonstration.

Label which policy a session is for with `--task`, which accepts any string:

```
conda run -n isaac_sim python collect_pickplace_demo.py --cube-start table --task pickup_policy --out ./raw_episodes
conda run -n isaac_sim python collect_pickplace_demo.py --place-target table2 --cube-start table2 --task place_policy --out ./raw_episodes
```

No new scene-setup mode was built for "place" episodes' already-holding-the-box starting
condition — it's achieved the same way every episode's starting condition always has been: jog the
robot into the hug pose (drive to the table, `U` to swing the arms around the box) *before*
pressing `B`, then only the approach→lower→release→retreat portion actually gets recorded.

**Recording boundaries matter now.** A `pickup` episode should end once the box is lifted and
you've backed away a step or two — press `B` to stop there, don't continue into a full carry to
the destination. A `place` episode should start already holding the box (from manually hugging it
into position before `B`) and end once released and backed away. This is different from how the
original data (still sitting in `raw_episodes/`, all labeled `pick_box_table_to_table2`/
`pick_box_table_to_cart`) was collected — those are one continuous pick-to-place trajectory with
no chassis motion at all, and are **21-dim**, incompatible with new 22-dim `pickup`/`place`
recordings (see next section). They can't be mixed.

## Step 1 — record episodes

Once Step 0 works, start a real session (pick one of the two `--task` examples above depending on
which policy you're recording for):

```
conda run -n isaac_sim python collect_pickplace_demo.py --cube-start table --task pickup_policy --deck-riser <tuned value> --out ./raw_episodes
```

Controls during recording:

| Key | Action |
|---|---|
| `W`/`S` | drive forward/back — **now part of the recorded action** (`chassis_forward`), used for the approach/retreat legs of each policy |
| `A`/`D`/`Q`/`E` | strafe/rotate — still manual-only, not recorded, use for repositioning/correction |
| `I`/`K` | torso up/down |
| `U`/`O` | swing both arms forward/back (the hug) |
| `J`/`L` | raise/lower both hands |
| `M`/`N` | close/open both grippers (optional) |
| `B` | toggle: start recording → stop and await a label |
| `Y` | (after stop) label the episode **success** and save it |
| `F` | (after stop) label the episode **failure** and save it |
| `Backspace` | (after stop) discard the episode, don't save |
| `R` | reset robot/cube/cart to spawn, re-randomizes the box's jittered pose (also discards an in-progress episode) |

Per episode: jog into position (for `place`, that means hugging the box first) → `B` (start) →
perform the policy's motion (see the pickup/place step lists above) → `B` (stop) →
`Y`/`F`/`Backspace` → `R` → repeat.

Episodes land in `./raw_episodes/episode_0000/`, `episode_0001/`, ... regardless of session, each
containing:

```
episode_0000/
  manifest.json           # success flag, fps, task name, joint names, frame count
  data.npz                # observation.state (T,22) and action (T,22) float32 arrays
  frames/NNNNNN_rgb.png   # RGB frame, one per recorded timestep
  frames/NNNNNN_depth.npy # raw depth (meters, float32, 480x640) for the same timestep -
                          # captured "just in case" a future policy wants it; not currently used
                          # by convert_to_lerobot.py (see that script's own docstring)
```

The 22nd `state`/`action` dim is `chassis_forward` — asymmetric, unlike every other dim: state is
the chassis's cumulative signed displacement (meters) along its own forward axis since the current
recording attempt started (0.0 at the first frame, resets every `B`-start/`R`-reset — not an
absolute world position), action is the forward/back drive *command* that tick (a velocity, not a
position target). See `CLAUDE.md` for the full derivation.

Depth also shows live in the browser viewer (a false-colored preview, same as `stream_demo.py`'s)
alongside RGB, purely for your own viewing convenience — the recorded depth is the raw float
array, not this colorized preview.

Record the reverse direction (or the other policy) in a second session — both share `--out`, so
episode numbering continues automatically, no manual bookkeeping needed. Aim for roughly **20-30+
successful episodes per policy** as a starting point; more, and more varied approach angles/box
poses, generally helps.

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
dependencies could conflict with Isaac Sim's pinned versions of numpy/torch/opencv/etc.). Full
setup, confirmed working end-to-end (not just "should work"):

```
conda create -n lerobot python=3.10 -y
conda activate lerobot
pip install lerobot
conda install -c conda-forge ffmpeg -y   # needed for reading the dataset back later (Step 5/6) -
                                          # video decoding fails without a matching-ABI ffmpeg,
                                          # even though conversion itself succeeds without it
```

Then convert (use a `--repo-id` that matches the `--task` you recorded with, e.g.
`local/pickup_policy` for `pickup_policy` episodes):

```
python convert_to_lerobot.py --raw-dir ./raw_episodes --repo-id local/pickup_policy --root ./lerobot_dataset
```

By default only episodes labeled **success** are included (`--include-failures` to add failures
too — they stay on disk either way, nothing is deleted). Verify the conversion actually worked,
not just that it exited 0 (video decode failures only surface on *read*, not on write):

```
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('local/pickup_policy', root='./lerobot_dataset')
print(ds.num_episodes, ds.num_frames, ds[0]['observation.state'].shape)
"
```

## Step 4 — train

Still in the `lerobot` environment, once converted (confirmed flags, `lerobot` 0.4.4 — check
`lerobot-train --help` on your installed version, this shifts across releases):

```
lerobot-train \
  --dataset.repo_id=local/pickup_policy \
  --dataset.root=./lerobot_dataset \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=./act_training/pickup_policy \
  --job_name=act_pickup_policy \
  --batch_size=8 \
  --num_workers=2 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

`--num_workers=2`/`--batch_size=8` are conservative defaults for a machine with limited system RAM
alongside a 12GB GPU — raise them if you have headroom. On an RTX 4080-class GPU this took ~1 hour
for 30k steps on a 35-episode dataset. Checkpoints land in `<output_dir>/checkpoints/NNNNNN/` plus
a `checkpoints/last` symlink to the most recent one.

Close Isaac Sim before training if you're tight on GPU memory — data collection and training don't
need to run at the same time.

## Step 5 — evaluate offline (no Isaac Sim needed)

Before investing in a live rollout test, sanity-check the checkpoint by replaying real recorded
episodes through its actual inference path and comparing predicted vs. recorded actions:

```
conda run -n lerobot python evaluate_act_checkpoint.py \
  --checkpoint-dir ./act_training/pickup_policy/checkpoints/last/pretrained_model \
  --raw-dir ./raw_episodes --dataset-root ./lerobot_dataset --dataset-repo-id local/pickup_policy
```

This is an **in-sample, open-loop** check — the policy is fed the *true* recorded observation at
every step, never what it would see after acting on its own prediction, so a low error here means
training converged, not that the policy can control the robot end-to-end. See the script's own
module docstring for details. Low/flat mean absolute error (a few hundredths of a radian) is a
good sign; a policy that can't even track its own training data isn't worth testing live.

## Step 6 — closed-loop rollout in Isaac Sim

Real validation needs the policy actually driving the robot, consuming its own predictions'
consequences frame after frame. This needs two processes (policy inference needs `torch`/
`lerobot`, which can't go in the `isaac_sim` env) talking over a local socket:

**Terminal 1 — start the policy server** (`lerobot` env):

```
conda run -n lerobot python policy_server.py \
  --checkpoint-dir ./act_training/pickup_policy/checkpoints/last/pretrained_model \
  --dataset-root ./lerobot_dataset --dataset-repo-id local/pickup_policy
```

**Terminal 2 — launch Isaac Sim in rollout mode** (`isaac_sim` env), matching whatever scene args
the checkpoint was trained on (`--place-target`/`--cube-start` etc.):

```
conda run -n isaac_sim python collect_pickplace_demo.py --place-target table2 --cube-start table --rollout
```

The robot connects to the policy server at startup (fails loudly if it's not running — this
doesn't fall back to teleop). `B`/`Y`/`F`/`Backspace`/`R` work exactly as in teleop mode: `B`
starts a rollout attempt (also resets the policy's internal state), the policy drives the arms/
torso/grippers/chassis-forward through the same safety clamps teleop uses, `Y`/`F`/`Backspace`
label it, `R` resets and re-randomizes the box pose for the next attempt. Attempts record to
`./rollout_episodes` by default (not `raw_episodes/` — these are policy predictions, not human
demonstrations, and shouldn't silently mix into training data).

**The checkpoint's dimension must match the current script's schema** — a 22-dim (`pickup_policy`/
`place_policy`) checkpoint queried correctly will work; feeding it a 21-dim state (old-schema data)
or vice versa fails loudly (a shape-mismatch crash), not silently. Don't mix them.

## Quick reference: file map

| File | Runs in | Purpose |
|---|---|---|
| `collect_pickplace_demo.py` | `isaac_sim` conda env | Scene + keyboard teleop + episode recorder; `--rollout` for closed-loop policy control |
| `check_raw_episodes.py` | plain `python3` | Validates recorded episodes before conversion |
| `convert_to_lerobot.py` | `lerobot` env | Raw episodes → LeRobotDataset |
| `evaluate_act_checkpoint.py` | `lerobot` env | Offline, open-loop action-tracking check on a trained checkpoint |
| `lerobot_policy_utils.py` | `lerobot` env | Shared checkpoint-loading helper (used by the two files above and `policy_server.py`) |
| `policy_server.py` | `lerobot` env | Inference server for `--rollout` mode - loads a checkpoint, serves predictions over a local socket |
| `policy_client.py` / `policy_wire.py` | `isaac_sim` env (stdlib + numpy only) | Client + wire protocol `collect_pickplace_demo.py --rollout` uses to talk to `policy_server.py` |
