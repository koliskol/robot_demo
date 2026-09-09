"""Drive a Galbot G1 mobile robot through a table<->pushcart pick-and-place task in Isaac Sim,
recording keyboard-teleoperated demonstrations to disk for later conversion into a LeRobot
dataset (see convert_to_lerobot.py, run in a separate lerobot-installed environment - this
script deliberately never imports lerobot itself, to keep it out of the isaac_sim conda env's
dependency footprint).

Usage:

    conda run -n isaac_sim python collect_pickplace_demo.py
    conda run -n isaac_sim python collect_pickplace_demo.py --out ./raw_episodes --deck-riser 0.5

Optionally open http://<host>:<port>/ (default http://0.0.0.0:8080/, see --host/--port) in a
browser and click Connect to watch the robot's front camera (RGB + false-colored depth preview)
live via WebRTC while teleoperating, same server as stream_demo.py (streaming_server.py) but its
own dedicated page (no map/point-cloud - this task has no lidar/world-state data). The page also
shows a recorder status badge and Start/Stop/Success/Fail/Discard buttons mirroring the
B/Y/F/Backspace keys below - either the browser buttons or the keyboard work interchangeably,
both drive the same recorder state machine.

Controls (viewport window must have focus) - drive/jog controls are unchanged from
stream_demo.py/../Robot_project/capture_cube_rgbd.py:

    W / S       drive forward / backward       (only needed to park/reposition between episodes -
    A / D       strafe left / right             the task itself is fixed-base: no driving is
    Q / E       rotate left / right             recorded as part of an episode's action space)
    I / K       hold to move torso up / down (leg lift joints)
    U / O       hold to swing both arms forward / back out to open (shoulder joint only) -
                this is the hug motion: swinging forward compresses the box between the forearms
    J / L       hold to raise / lower both hands (elbow joint only)
    M / N       hold to close / open both grippers (optional - the hug, not the gripper, is the
                primary hold; fingers can add a little extra contact but aren't required)
                In --rollout mode, grippers are policy-controlled instead, so M/N are repurposed:
                press M to activate the pickup policy (--policy-host/--policy-port, task=--task),
                press N to activate the place policy (--policy2-host/--policy2-port, task=--task2).
                Each also toggles: pressing the same key again while that policy is the one
                currently running stops the attempt (same transition as B) so it can be labeled
                with Y/F/Backspace - there is no learned "done" signal, the policy predicts actions
                forever otherwise, so this is the only way to end an attempt. Pressing the *other*
                key while one is already running just switches which policy is active without
                stopping the recording. Two separate policy_server.py processes must be running,
                one per checkpoint/port. Switching or restarting resets the newly-activated
                policy's internal action-chunk queue.
    C / V       hold to rotate both wrists on joint5 (link7's local X - confirmed by the user as
                the correct rotation to fix the gripper's diagonal resting angle; see
                WRIST_X_JOINT_INDEX's comment). Defaults to the live-calibrated flat/horizontal
                pose (STARTING_WRIST_X_RAD = -0.7400 rad / -42.4deg) at launch/reset, so C/V are
                only needed for further fine adjustment - prints "[wrist] joint5=...deg" on
                release if you do.
    , / .       hold to rotate both wrists on joint6 (link7's local Y) - re-enabled alongside C/V
                to try a different grab approach. No live-calibrated default yet
                (STARTING_WRIST_Y_RAD = 0) - prints "[wrist] joint6=...deg" on release.
    [ / ]       hold to rotate both wrists on joint7 (link7's local Z, by definition - its own
                <axis> in the URDF IS that link's Z). No live-calibrated default yet
                (STARTING_WRIST_Z_RAD = 0) - prints "[wrist] joint7=...deg" on release.
                In --rollout mode all three of these are policy-controlled, so C/V/,/./[/] are
                inert there.
    R           reset the robot/cube/cart to spawn pose (also discards any in-progress episode)

    B           toggle: start recording an episode / stop and await a label
    Y           (after B-stop) label the just-recorded episode a SUCCESS and save it
    F           (after B-stop) label the just-recorded episode a FAILURE and save it
    BACKSPACE   (after B-stop) discard the just-recorded episode without saving

    Close the viewport window to exit.

IMPORTANT - read before collecting any real data: this task requires holding and carrying a loose
object, which neither this project nor ../Robot_project/capture_cube_rgbd.py has ever
demonstrated. The chosen approach is a bimanual "hug" - both arms swinging forward (U) to
compress the box between the forearms, rather than a single gripper's fingertip pinch - so the
boxes (real warehouse cardboard-box assets, see BOX_ASSET_MAIN/CUBE2/CUBE3 below - not procedural
cubes) are
sized bigger than a gripper-sized grasp would need. This is still friction-only contact, same
constraint as a gripper pinch would have been: ../Robot_project/capture_cube_rgbd.py's own
history records that every *kinematic* grasp-assist attempt (a hand-authored FixedJoint, and
Isaac Sim's own IsaacSurfaceGripper) reproducibly destabilized the whole robot when attached to a
driven articulation link, since it's actively driven by the articulation's own solver rather than
a simple independently-jointed body. Do not add any joint-based/kinematic attach mechanism to
"help" the hug hold - if it isn't stable on the assets' own baked-in friction alone (plus arm
swing compression), the fix is box mass/scale and swing-in distance (or, as a next step, a custom
high-friction PhysicsMaterial bound onto the box - not yet done here, see the comment above
spawn_real_box), not a new attach primitive. Before recording anything, manually jog through one
full pick-table / place-cart / pick-cart / place-table cycle and confirm the hug is physically
stable and both arms can actually converge around the box from a single parked pose - see
PUSHCART_DECK_HALF_EXTENT / --deck-riser / ROBOT_APPROACH_GAP_M below for the other knobs to turn
if the geometry doesn't work on the first try.

The arm/hand/torso jog constants mirror stream_demo.py and ../Robot_project/capture_cube_rgbd.py
exactly (same Galbot G1 asset, same joint targets/clamps/rates) - see stream_demo.py's module
docstring for the full derivation. The camera mount is NOT the same as either sibling script's -
this one is head-mounted, not chassis-mounted, see HEAD_CAMERA_MOUNT's comment for why. This
script drops lidar entirely
(not needed for offline data collection) but does capture depth (RGB + depth are both recorded,
"just in case" a future policy wants it - see EpisodeRecorder.save; nothing in convert_to_lerobot.py
uses it yet, that's a deliberately-unimplemented next step, see that script's own comment) and
adds a pushcart + graspable boxes, ported/adapted from capture_cube_rgbd.py's build_pushcart and
cube spawn (see those functions below for what changed and why). It reuses streaming_server.py's
video tracks (same as stream_demo.py) purely for live viewing convenience - streamed frames are
NOT what gets recorded to disk; recording samples at a fixed rate via EpisodeRecorder below,
independent of the WebRTC feed.
"""

import argparse
import json
from enum import Enum, auto
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--host", type=str, default="0.0.0.0", help="WebRTC viewing server bind address.")
parser.add_argument("--port", type=int, default=8080, help="WebRTC viewing server port.")
parser.add_argument("--out", type=str, default="./raw_episodes", help="Output directory for recorded episodes. Defaults to ./rollout_episodes instead when --rollout is set.")
parser.add_argument("--record-fps", type=float, default=15.0, help="Fixed sample rate for recorded episodes.")
parser.add_argument(
    "--rollout",
    action="store_true",
    default=False,
    help="Closed-loop rollout mode: instead of driving the arms/torso/grippers from held keys, "
    "query a running policy_server.py over --policy-host/--policy-port at --record-fps and apply "
    "its predicted joint targets (through the same clamp_to_actual() safety clamps teleop uses). "
    "B/Y/F/Backspace/R still work exactly as in teleop mode - B also resets the policy's internal "
    "state for a fresh attempt. Base driving (WASD/QE) still works for manual repositioning; the "
    "policy only ever controls the 21 recorded arm/torso/gripper dims, matching CLAUDE.md's note "
    "that this task has no base pose in its action space. See CLAUDE.md's \"LeRobot pick-and-place "
    "pipeline\" for why this is a socket client, not an in-process policy load - unverified live "
    "(no way to visually confirm a hug succeeds from a non-interactive session), re-check the "
    "Stage 0-style reach cycle by eye before trusting it.",
)
parser.add_argument("--policy-host", type=str, default="127.0.0.1", help="policy_server.py host for the pickup policy (M key), only used with --rollout.")
parser.add_argument("--policy-port", type=int, default=8765, help="policy_server.py port for the pickup policy (M key), only used with --rollout.")
parser.add_argument("--policy2-host", type=str, default="127.0.0.1", help="policy_server.py host for the place policy (N key), only used with --rollout.")
parser.add_argument("--policy2-port", type=int, default=8766, help="policy_server.py port for the place policy (N key), only used with --rollout.")
parser.add_argument("--task", type=str, default=None, help="Task name stored in each episode's manifest (default: derived from --cube-start). Also the task label sent to the pickup policy (M key) during --rollout.")
parser.add_argument("--task2", type=str, default="place_policy", help="Task label sent to the place policy (N key) during --rollout - independent of --task, which covers the pickup policy in this mode.")
parser.add_argument(
    "--place-target",
    type=str,
    choices=["cart", "table2"],
    default="cart",
    help="Which secondary object the scene builds as the table's pick/place partner - the "
    "pushcart (default, existing behavior) or a second table placed adjacent to the main one "
    "(see TABLE2_GAP_M). Only one is ever built per session - they're alternative task variants, "
    "not simultaneous targets (a single parked robot pose can't reach both at once).",
)
parser.add_argument(
    "--table2-jitter-m",
    type=float,
    default=0.05,
    help="Max random xy offset (meters) applied to table2's spawn/reset position (only when "
    "--place-target=table2), sampled fresh each episode same as --box-jitter-m - so place_policy "
    "sessions see the destination table at a different spot each attempt, not always the exact "
    "same place. 0 disables it. Ignored for --place-target=cart (not implemented there yet).",
)
parser.add_argument(
    "--table2-yaw-jitter-deg",
    type=float,
    default=5.0,
    help="Max random yaw offset (degrees) applied to table2's spawn/reset orientation, uniform in "
    "[-value, +value]. Kept smaller than --box-yaw-jitter-deg by default - a full table rotating "
    "meaningfully changes the reach geometry far more than a small box does. 0 disables it.",
)
parser.add_argument(
    "--cube-start",
    type=str,
    choices=["table", "cart", "table2"],
    default="table",
    help="Where the cube spawns on reset - run one session per direction to collect both "
    "directions (table->X and X->table). Must be 'table' or match --place-target (e.g. "
    "--cube-start cart requires --place-target cart).",
)
parser.add_argument(
    "--cube-scale",
    type=float,
    default=1.0,
    help="Uniform scale multiplier for the main box (a real warehouse cardboard-box asset, see "
    "BOX_ASSET_MAIN - native footprint is roughly 0.38 x 0.25 x 0.15m at scale 1.0, small enough "
    "to also fit the pushcart deck).",
)
parser.add_argument(
    "--cube-mass",
    type=float,
    default=0.15,
    help="Main box mass in kg - overrides the asset's own authored mass. Kept light-to-moderate "
    "since the hold is friction-only (arm compression), not a joint-based attach.",
)
parser.add_argument(
    "--cube-scale-min",
    type=float,
    default=None,
    help="If set together with --cube-scale-max, randomize the main box's scale uniformly in "
    "[min, max] on every spawn/reset instead of the fixed --cube-scale - real recorded sessions so "
    "far all used one fixed scale per session, so the trained policy has never seen box-size "
    "variation. Opt-in and unset by default so existing workflows are unaffected. Unlike the "
    "position/yaw jitter above, this respawns the box prim from scratch each reset (delete + "
    "re-reference + re-place), since place_on_surface's scale-then-measure trick only gives the "
    "right footprint at a prim's just-referenced identity transform - not yet live-verified, "
    "re-run the Stage 0 reach/hug cycle across your chosen range before trusting it for real "
    "collection.",
)
parser.add_argument(
    "--cube-scale-max",
    type=float,
    default=None,
    help="See --cube-scale-min.",
)
parser.add_argument(
    "--cube-scale-cycle",
    action="store_true",
    help="Requires --cube-scale-min/--cube-scale-max. Instead of sampling the main box's scale "
    "uniformly at random, step deterministically through small (--cube-scale-min) -> medium "
    "(the midpoint) -> big (--cube-scale-max) -> back to small, advancing one step on every "
    "spawn/reset (including the very first spawn) rather than drawing at random. Combine with "
    "--no-extra-boxes if you want exactly one box on screen at a time.",
)
parser.add_argument(
    "--box-jitter-m",
    type=float,
    default=0.03,
    help="Max random xy offset (meters) applied to the main box's spawn/reset position, sampled "
    "fresh each episode (uniform over a disk of this radius, not a square). Previously the box "
    "spawned at the exact same point every episode, which risks a policy that only ever learned "
    "one pixel-perfect box pose. Keep this modest and re-check the Stage 0 manual reach cycle at "
    "the extremes before widening it - ARM_FORWARD_POSE's hug convergence is tuned for one "
    "specific position and hasn't been verified to hold up from an arbitrarily far-off start. "
    "0 disables position jitter.",
)
parser.add_argument(
    "--box-yaw-jitter-deg",
    type=float,
    default=10.0,
    help="Max random yaw offset (degrees) applied to the main box's spawn/reset orientation, "
    "sampled fresh each episode uniformly in [-value, +value]. Same reach-margin caution as "
    "--box-jitter-m. 0 disables yaw jitter.",
)
parser.add_argument(
    "--starting-wrist-x-rad",
    type=float,
    default=None,
    help="Override STARTING_WRIST_X_RAD (default -0.7400, the live-calibrated flat/horizontal "
    "gripper pose) for this session's launch/reset starting pose and --rollout's pre-first-"
    "prediction fallback pose. Exists because this default was added 2026-09-08, AFTER "
    "pickup_policy/place_policy's episodes finished recording (through 2026-09-04) - every frame "
    "of that data has joint5 (wrist X) exactly 0.0, since no wrist control existed yet at the "
    "time (confirmed: ARM_OPEN_POSE/ARM_FORWARD_POSE are both 0.0 at that index, and nothing else "
    "touched it back then). Pass --starting-wrist-x-rad 0.0 when collecting more episodes meant to "
    "extend one of those existing datasets, so new episodes start from the same pose distribution "
    "as the old ones instead of silently drifting - leave unset (uses the current calibrated "
    "default) for any new/independent data collection.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Seed for the box spawn-jitter RNG (--box-jitter-m/--box-yaw-jitter-deg). Default: a "
    "fresh random sequence each run.",
)
parser.add_argument(
    "--cube2-scale",
    type=float,
    default=1.0,
    help="Scale multiplier for a second, bigger box (a distinct real cardboard-box asset, see "
    "BOX_ASSET_CUBE2 - native footprint is roughly 0.50 x 0.50 x 0.25m at scale 1.0). Table-side only.",
)
parser.add_argument("--cube2-mass", type=float, default=0.25, help="Mass of the second box in kg.")
parser.add_argument(
    "--cube3-scale",
    type=float,
    default=1.0,
    help="Scale multiplier for a third, even bigger box (a distinct real cardboard-box asset, see "
    "BOX_ASSET_CUBE3 - native footprint is roughly 0.70 x 0.50 x 0.50m at scale 1.0, the biggest). "
    "Table-side only.",
)
parser.add_argument("--cube3-mass", type=float, default=0.35, help="Mass of the third box in kg.")
parser.add_argument(
    "--extra-boxes",
    dest="extra_boxes",
    action="store_true",
    default=True,
    help="Spawn the two extra boxes (default: on). Only appear when --cube-start=table - the "
    "pushcart deck is too small to fit 3 boxes side by side.",
)
parser.add_argument("--no-extra-boxes", dest="extra_boxes", action="store_false", help="Spawn only the single primary box.")
parser.add_argument(
    "--table-height-scale",
    type=float,
    default=0.69,
    help="Height-only scale factor for the main table (legs shortened, tabletop footprint "
    "unchanged - see place_on_ground's z_scale param). Default lowers table_top_z from the "
    "native ~0.72m to ~0.5m, for easier arm reach. 1.0 = native height. Re-check --deck-riser "
    "via the Stage 0 manual reach check after changing this - it changes the table/cart height "
    "delta printed in the startup [geometry] diagnostic.",
)
parser.add_argument(
    "--deck-riser",
    type=float,
    default=0.0,
    help="Extra meters added to the pushcart deck's stock height (~0.15m) - tune this after the Stage 0 "
    "manual reach check described in the module docstring; raise it if the gripper can't get low enough "
    "over the deck to release the cube.",
)
parser.add_argument(
    "--drive-speed",
    type=float,
    default=0.4,
    help="Chassis drive/strafe command magnitude. Lower than stream_demo.py's 1.0 default - at "
    "1.0, live-observed the robot tipping over when driving fast, especially on diagonal "
    "drive+strafe combos (command magnitude adds, can exceed 1.0). Only affects parking/"
    "repositioning between episodes, not anything recorded.",
)
parser.add_argument(
    "--turn-speed",
    type=float,
    default=0.08,
    help="Chassis rotation command magnitude. Lower than stream_demo.py's 0.3 default, same "
    "tip-over reasoning as --drive-speed - lowered further from an original 0.15 after live "
    "testing showed rotating (Q/E) while gripping the pushcart handle could release/break the "
    "grip: a fast chassis rotation swings the whole robot body around the fixed grip point, "
    "generating a large reaction torque through the arm into the handle. A slower rotation gives "
    "the arm's compliance/lead-clamp mechanisms (see JOINT_EFFORT_LIMIT_NM/ARM_CONTACT_MAX_LEAD_RAD) "
    "more time to track the changing geometry instead of being wrenched. UNVERIFIED LIVE (no "
    "Isaac Sim available while writing this) - re-test grip-then-rotate at this value; lower "
    "further if the grip still releases, since rotation speed alone may not be the whole story.",
)
parser.add_argument("--arm-speed", type=float, default=0.4, help="Max arm joint speed in radians/second.")
parser.add_argument("--torso-speed", type=float, default=0.4, help="Torso up/down speed, as a fraction/second of its full travel.")
args = parser.parse_args()
if args.cube_start not in ("table", args.place_target):
    parser.error(f"--cube-start {args.cube_start!r} requires --place-target {args.cube_start!r} (got --place-target {args.place_target!r})")
if (args.cube_scale_min is None) != (args.cube_scale_max is None):
    parser.error("--cube-scale-min and --cube-scale-max must be given together.")
if args.cube_scale_min is not None and not (0.0 < args.cube_scale_min <= args.cube_scale_max):
    parser.error(f"--cube-scale-min ({args.cube_scale_min}) must be > 0 and <= --cube-scale-max ({args.cube_scale_max}).")
if args.cube_scale_cycle and args.cube_scale_min is None:
    parser.error("--cube-scale-cycle requires --cube-scale-min/--cube-scale-max to also be set.")
if args.task is None:
    args.task = f"pick_box_table_to_{args.place_target}" if args.cube_start == "table" else f"pick_box_{args.place_target}_to_table"
if args.rollout and args.out == parser.get_default("out"):
    # Rollout attempts are policy predictions, not human demonstrations - default them into a
    # separate directory so they never silently mix into raw_episodes/ training data. An explicit
    # --out still overrides this, e.g. to intentionally fold graded rollouts back in later.
    args.out = "./rollout_episodes"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow
import omni.graph.core as og
import isaacsim.core.utils.bounds as bounds_utils
from PIL import Image
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from streaming_server import FrameStore, run_in_background
from policy_client import PolicyClient

TABLE_ASSET = "/Isaac/Environments/Office/Props/SM_TableB.usd"
ROBOT_ASSET = "/Isaac/Robots/Galbot/galbot_g1/galbot_g1.usda"
ROBOT_PRIM = "/World/Robot"

# Camera is head-mounted, not chassis-mounted like stream_demo.py's front_camera - this asset
# has a real 2-DOF head (Head_Golf: head_joint1/head_joint2) with its own purpose-built sensor
# mount point, head_end_effector_mount_link, found by walking the robot's full prim tree live
# (not guessed). Nothing in this project drives the head joints, so it just sits at its rest
# pose - but a head-mounted camera still moves with torso crouch (I/K), unlike a chassis-mounted
# one, which stays at a fixed height/angle regardless of torso pose. That coupling cuts both
# ways: live-observed a large dark shape filling most of the frame when the torso is crouched
# most of the way down (K held near max) *and* the arms are swung forward at the same time - this
# is the robot's OWN torso/shoulder self-occluding the head-mounted camera's view, not a clipping
# bug (confirmed via a controlled test: driving torso+arms to their crouched/forward poses under
# the same clamp_to_actual scheme this file's main loop uses, not letting them free-fall-settle,
# then rendering - the frame is dominated by the robot's own body, matching what was reported
# live). Tilting the camera down further does NOT fix this - it was tested and made the robot's
# own torso fill even more of the frame, not less, since the torso is what's now closest to the
# lens once crouched this far. If the hand still isn't visible with the tilt below, try less
# torso crouch (partial I/K) relying more on elbow lift (J/L) to keep the torso from being what's
# directly in front of the camera - not something this file's constants alone can fix, since it
# depends on how you jog the robot live.
HEAD_CAMERA_MOUNT = (
    f"{ROBOT_PRIM}/OmniChassis/base_link/omni_chassis_base_link/omni_chassis_leg_mount_link/leg_base_link/"
    "leg_link1/leg_link2/leg_link3/leg_link4/leg_link5/leg_end_effector_mount_link/torso_base_link/Head_Golf/"
    "torso_base_link/torso_head_mount_link/head_base_link/head_link1/head_link2/head_end_effector_mount_link"
)

# head_joint1/head_joint2's own PhysX joint drive (separate from HEAD_CAMERA_MOUNT above, which
# is the sensor mount several links further out - these are the actual 2-DOF pan/tilt joints)
# ships with very weak stiffness/damping (~2.8/0.001 and ~0.99/0.0004 respectively, confirmed
# live) and, unlike every other controlled joint group in this file, nothing ever commands them -
# live-tested that this lets the head swing up to ~36deg during ordinary torso+arm motion (a 1s
# full-range torso crouch + arm swing cycle, well within what teleoperating this robot actually
# does), which is what causes the camera view to visibly tilt in unintended directions - not a
# bug in HEAD_CAMERA_MOUNT's roll/tilt math (confirmed separately, and correct, via static
# rendering). See stiffen_head_joints()/hold_head_joints() below for the two-part fix.
HEAD_JOINT_PATHS = [
    f"{ROBOT_PRIM}/OmniChassis/base_link/omni_chassis_base_link/omni_chassis_leg_mount_link/leg_base_link/"
    f"leg_link1/leg_link2/leg_link3/leg_link4/leg_link5/leg_end_effector_mount_link/torso_base_link/Head_Golf/"
    f"joints/head_joint{i}"
    for i in (1, 2)
]


def stiffen_head_joints() -> None:
    """Raise the head joints' PhysX drive stiffness/damping from their very weak defaults, before
    the articulation is initialized. Live-tested under the same aggressive 1s-cycle stress test
    referenced above: stiffness/damping 2.8/0.001 -> 200/20 cut max drift from ~36deg (uncommanded)
    / ~26deg (commanded to hold 0 but with weak drive) down to ~12deg; going stiffer still
    (2000/100) barely helped further (~11deg) - diminishing returns confirmed, not pushed beyond
    200/20. This alone doesn't fully solve the wobble - combine with hold_head_joints() below,
    called every frame in the main loop, same as every other controlled joint group.
    """
    for path in HEAD_JOINT_PATHS:
        drive = UsdPhysics.DriveAPI.Get(get_prim_at_path(path), "angular")
        drive.GetStiffnessAttr().Set(200.0)
        drive.GetDampingAttr().Set(20.0)


def hold_head_joints(robot: SingleArticulation, head_dof_indices: list) -> None:
    """Command the head joints to hold their rest pose (0, 0) - call once per physics step, same
    as every other controlled joint group's per-frame apply_action. No lead clamp here unlike
    arm/gripper: live-tested it makes no measurable difference (the head has nothing to hit/push
    against, so there's no contact-instability reason to cap correction speed the way there is
    for the arm/gripper)."""
    robot.apply_action(ArticulationAction(joint_positions=np.zeros(2), joint_indices=head_dof_indices))


def stiffen_gripper_friction(stage, robot_prim: str) -> None:
    """Bind a higher-friction PhysicsMaterial (see GRIPPER_FRICTION_COEFF's comment for why this,
    not a stronger pinch, is the right lever) to the gripper's real finger/knuckle link prims and
    their "collisions" sub-geometry. Call once, before world.reset(), same as stiffen_head_joints()
    above.

    First version of this function matched by keyword ("gripper"/"finger"/"knuckle" as a path
    substring) AND required prim.HasAPI(UsdPhysics.CollisionAPI) - confirmed live to match zero
    prims. dump_gripper_hierarchy() (see above) then found the real names via the drive joint's
    body0/body1 targets: gripper_{l,r}_finger_link, gripper_{l,r}_knuckle_link, gripper_{l,r}_
    inner_knuckle_link, each with their own "visuals"/"collisions" sub-prims (e.g. .../gripper_l_
    finger_link/collisions/link_3). That same dump also showed HasAPI(UsdPhysics.CollisionAPI)
    reads False on literally every prim in the hand assembly, including ones named "collisions/
    link_N" - i.e. that check is not a reliable collision-prim filter for this asset (collision is
    authored some other way it doesn't detect). This version drops that check entirely and matches
    the real link names directly instead - binding a material to a prim that turns out not to be
    an actual collider is harmless (no effect), so casting this net is strictly safer than the
    keyword-plus-HasAPI combination that provably missed everything.

    UNVERIFIED LIVE (no Isaac Sim available while writing this) - read the printed prim list on
    launch to confirm it's non-empty and look like the finger surfaces; there's still no live
    confirmation the friction value itself actually changes contact behavior, since HasAPI can't
    be used here to confirm these are real colliders ahead of time.
    """
    material = UsdShade.Material.Define(stage, "/World/GripperFrictionMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(GRIPPER_FRICTION_COEFF)
    material_api.CreateDynamicFrictionAttr(GRIPPER_FRICTION_COEFF)
    material_api.CreateRestitutionAttr(0.0)

    keywords = ("finger_link", "knuckle_link", "/collisions/")
    bound_paths = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim)):
        path_str = str(prim.GetPath()).lower()
        if any(keyword in path_str for keyword in keywords):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
            )
            bound_paths.append(str(prim.GetPath()))

    if bound_paths:
        print(f"[gripper] bound friction={GRIPPER_FRICTION_COEFF} to {len(bound_paths)} prim(s):")
        for path in bound_paths:
            print(f"    {path}")
    else:
        print(
            "[warning] stiffen_gripper_friction matched no prims under "
            f"{robot_prim!r} for keywords {keywords} - friction material NOT applied anywhere. "
            "Check the actual gripper/finger prim names live and adjust the keyword list."
        )


def dump_arm_joint_limits(stage, robot_prim: str) -> None:
    """One-time diagnostic dump (not a fix) - print every arm joint's actual authored PhysX
    revolute-joint limits and drive stiffness/damping, straight from the live asset, before this
    file changes anything. Added after a pinch-the-handle-then-drive test drove left_arm_joint5/6
    to 14.79/7.37 rad - several full rotations past any plausible limit - which raised a real
    question this file has never actually checked: whether these joints have a PhysX rotation
    limit enforced at all in THIS asset, as opposed to the separate offline reference URDF
    (../Robot_project/urdf/galbot_g1_*_arm.urdf) that ARM_FORWARD_POSE/WRIST_X_MIN_RAD/etc.
    were derived from - a USD asset's actual joint authoring doesn't have to match a
    separately-generated URDF just because both describe the same robot. Call once, right after
    add_reference_to_stage, before any of this file's own stiffen_*/hold_* joint modifications.
    """
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim)):
        path_str = str(prim.GetPath())
        if "_arm_joint" not in path_str:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        stiffness = drive.GetStiffnessAttr().Get() if drive else None
        damping = drive.GetDampingAttr().Get() if drive else None
        print(f"[joint-limits] {path_str}: lower={lower} upper={upper} stiffness={stiffness} damping={damping}")


def dump_gripper_hierarchy(stage, robot_prim: str) -> None:
    """One-time diagnostic dump (not a fix) - find the real prim paths of the gripper's internal
    finger/knuckle links, straight from the live asset, since guessing them has already failed
    once: stiffen_gripper_friction's keyword match ("gripper"/"finger"/"knuckle" as a path
    substring) found zero collision prims, meaning this asset names those parts something else
    entirely - this file has never actually enumerated them, only the drive joint's DOF name
    (left_gripper_joint/right_gripper_joint, used via robot.get_dof_index()) was ever confirmed.

    Finds the drive joint prim itself the same reliable way dump_arm_joint_limits found the arm
    joints (path suffix match, not a keyword guess - a joint's own name is not in question, only
    its surrounding links are), reads its body0/body1 targets directly (the actual two links it
    connects), then walks and prints every descendant of the joint's parent prim - the whole hand
    assembly - along with which ones carry CollisionAPI, so the real finger/knuckle names (and
    which of them actually need the friction material) are known instead of guessed.
    """
    for side in ("left", "right"):
        joint_name = f"{side}_gripper_joint"
        joint_prim = None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim)):
            if str(prim.GetPath()).endswith(joint_name):
                joint_prim = prim
                break
        if joint_prim is None:
            print(f"[gripper-hierarchy] no prim found ending in {joint_name!r} under {robot_prim!r}")
            continue

        joint = UsdPhysics.Joint(joint_prim)
        body0 = joint.GetBody0Rel().GetTargets() if joint else []
        body1 = joint.GetBody1Rel().GetTargets() if joint else []
        print(f"[gripper-hierarchy] {joint_name} prim: {joint_prim.GetPath()}")
        print(f"[gripper-hierarchy] {joint_name} body0={body0} body1={body1}")

        # joint_prim.GetParent() alone lands on the "joints" subfolder (siblings of joint_prim,
        # all correctly collision=False, no actual link geometry there - confirmed live, this bug
        # was in the first version of this function) - the real link/mesh tree (e.g. gripper_
        # flange, gripper_r_knuckle_link per body0/body1 above) is one level further up, as a
        # sibling of "joints".
        hand_root = joint_prim.GetParent().GetParent()
        print(f"[gripper-hierarchy] {side} hand assembly under: {hand_root.GetPath()}")
        for prim in Usd.PrimRange(hand_root):
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            print(f"    {prim.GetPath()}  (collision={has_collision})")


# arm_joint5/6/7 (the wrist) are barely driven by this file's own control scheme - joint7 is
# always commanded to 0 (nothing ever adds to that index); joint5/6 (see WRIST_X/Y_JOINT_INDEX)
# each only get a modest fixed position target. Live-observed via the
# watchdog (see GRIPPER_WATCHDOG_VELOCITY_RAD_S) during a pinch-the-handle-then-move test:
# right_arm_joint7 reached +2.2657 rad - past its own documented hard limit of +-1.5382 rad - at
# 3.19 rad/s, and right_arm_joint5 was dragged from its -0.74 rad target out to -1.756 rad at
# -75 Nm of measured effort. This is the same class of problem already found and fixed once in
# this file for the head joints (HEAD_JOINT_PATHS/stiffen_head_joints): a joint with weak native
# PhysX drive stiffness holds its commanded position fine with no load, but gets shoved far off
# target by a real external reaction force (here, from gripping the handle and then driving) that
# the weak drive can't resist - the tight lead clamp on the *target* doesn't help, since it's the
# *actual* position being dragged off course, not the target running ahead of it.
WRIST_JOINT_STIFFNESS = 2000.0
WRIST_JOINT_DAMPING = 200.0


def stiffen_wrist_joints(stage, robot_prim: str) -> None:
    """Raise arm_joint5/6/7's (both arms) PhysX drive stiffness/damping from their native
    defaults, before the articulation is initialized - same technique and call-site timing as
    stiffen_head_joints() above, applied to a different weak-drive joint group found via the
    watchdog instead of the earlier head-wobble live test.

    Discovers the actual joint prims by walking the live prim tree and matching path suffixes
    (this file has never enumerated arm_joint5/6/7's own USD prim paths - only their DOF names,
    used via robot.get_dof_index()) rather than a hardcoded guessed path, printing every prim it
    stiffens so this can be checked live rather than trusted blind.

    UNVERIFIED LIVE (no Isaac Sim available while writing this) - 2000/200 is a scaled-up guess
    from stiffen_head_joints()'s live-tuned 200/20 (same 10:1 stiffness:damping ratio, scaled up
    for the arm's larger mass/torques vs. the head), not itself live-tested. Re-run the exact
    pinch-the-handle-then-move test that surfaced this and confirm the watchdog stays quiet;
    raise further (same diminishing-returns pattern stiffen_head_joints() found - try roughly
    10x before assuming it's not helping) if it still trips, or lower it if it introduces new
    stiffness-related jitter/oscillation that wasn't there before.
    """
    keywords = ("left_arm_joint5", "left_arm_joint6", "left_arm_joint7", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7")
    stiffened_paths = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim)):
        path_str = str(prim.GetPath())
        if any(path_str.endswith(keyword) for keyword in keywords):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            drive.GetStiffnessAttr().Set(WRIST_JOINT_STIFFNESS)
            drive.GetDampingAttr().Set(WRIST_JOINT_DAMPING)
            stiffened_paths.append(path_str)

    if stiffened_paths:
        print(f"[wrist] stiffened {len(stiffened_paths)} joint(s) to {WRIST_JOINT_STIFFNESS}/{WRIST_JOINT_DAMPING}:")
        for path in stiffened_paths:
            print(f"    {path}")
    else:
        print(
            "[warning] stiffen_wrist_joints matched no joint prims under "
            f"{robot_prim!r} for keywords {keywords} - wrist stiffness NOT changed anywhere. "
            "Check the actual arm_joint5/6/7 prim names live and adjust the keyword list."
        )


# The mount link's own local frame does not face the robot's forward direction - confirmed live
# by rendering at identity orientation (showed a sideways, rolled view, not forward) and testing
# candidate corrections: a +90deg rotation about local X (CAMERA_ROLL_DEG) is what re-aligns it,
# tested by rendering and visually confirming a normal-looking horizon. CAMERA_TILT_DEG is then a
# small *additional* downward pitch on top of that correction, composed via quat_multiply (pitch
# applied after roll - see camera_head_mount_quat) - much smaller than a chassis mount would need
# (10deg, not 45deg) because the head sits much higher and further forward than the chassis ever
# did, so the look-down angle to a table-height box is shallow, not steep; confirmed by computing
# the actual head-to-box world vector live rather than guessing. CAMERA_FOV_DEG is widened from
# stream_demo.py's 60deg to help keep a nearby box in frame. All of this confirmed working for
# the box-on-table view at the robot's normal ~0.9m parked distance.
CAMERA_ROLL_DEG = 90.0
CAMERA_TILT_DEG = 26.7
CAMERA_FOV_DEG = 90.0

# Runtime camera pan/tilt calibration (LEFT/RIGHT/UP/DOWN keys, or the browser's rotate buttons -
# see camera_pan_tilt_quat below). CAMERA_TILT_DEG above is the starting tilt; CAMERA_PAN_DEG is
# the starting yaw (found via live calibration through this same mechanism, not the mount's
# untouched default - see HEAD_CAMERA_MOUNT's own comment on the roll derivation). Whenever a
# live session finds a better angle, the console prints the exact pan/tilt to paste back in here.
CAMERA_PAN_DEG = 22.0
CAMERA_ROTATE_KEY_SPEED_DEG_S = 20.0
CAMERA_ROTATE_STEP_DEG = 2.0  # per browser-button click

# A dark curved shape intruding into the lower part of the frame (live-reported, then confirmed
# via a physics raycast, not guessed) turned out to be part of the robot's OWN head housing
# (head_link2's own collision mesh) - the mount point sits close enough to the head's own shell
# that its lower edge pokes into the camera's own field of view. Pushing the camera forward along
# the mount's local +X clears it entirely (confirmed live: 0.0 shows the obstruction, +0.1 fully
# clears it, -0.1 makes it fill most of the frame instead - direction confirmed both ways, not
# just tested one way and assumed).
CAMERA_MOUNT_FORWARD_OFFSET_M = 0.1

# Isaac Sim's Camera defaults to a 1.0m NEAR clipping plane (confirmed live via
# camera.get_clipping_range() - not a documented default anyone would guess) - anything closer
# than that to the lens is silently not rendered at all, which is almost certainly why a box or
# the robot's own hand "disappeared" once brought close during the hug: the box-on-table view
# above never gets that close (robot stays parked ~0.9m back), but the hug itself absolutely
# does. Confirmed the fix directly: with the default 1.0m clip, an object placed ~0.4m from the
# lens rendered as nothing at all; with CAMERA_NEAR_CLIP_M applied, the same object is visible.
#
# 0.1, not smaller - a first attempt at 0.02 was live-tested and made the ENTIRE render go
# almost black (mean pixel brightness dropped from ~195 to ~0.15, confirmed via a sweep: 0.02 and
# 0.03 both broke it, 0.05 was still badly dark, 0.08 partially recovered, 0.1 and above matched
# normal baseline brightness exactly) - independent of the far value, so this isn't the usual
# near/far-ratio depth-precision story, more likely something specific to how the RTX renderer's
# auto-exposure or a similar pass reacts to a near-zero near plane. 0.1 is still a real
# improvement over the 1.0m default (confirmed a box at ~0.35m renders clearly at this setting)
# without triggering whatever breaks at smaller values - do not lower this without re-testing
# actual rendered brightness, not just whether the call succeeds.
# CAMERA_FAR_CLIP_M is just tightened from the 1,000,000m default to something matching this
# scene's actual scale - not itself part of the close-up fix, and not implicated in the above.
CAMERA_NEAR_CLIP_M = 0.1
CAMERA_FAR_CLIP_M = 50.0


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1*q2 (both (w,x,y,z)) - applying the result to a vector is equivalent to
    applying q2 first, then q1. Verified against scipy.spatial.transform.Rotation's composition
    before use (not just assumed correct)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def camera_head_mount_quat(roll_deg: float, tilt_deg: float) -> np.ndarray:
    """Composed correction quaternion for the head camera mount - see HEAD_CAMERA_MOUNT's comment
    for the derivation. Both component rotations use Camera.set_local_pose's default "world" axes
    convention (+Z up, +X forward, per that method's own docstring): a positive rotation about X
    is the roll correction, a positive rotation about Y (applied second, i.e. in the roll-
    corrected frame) tilts the look direction down.
    """
    roll_half = np.radians(roll_deg) / 2.0
    roll_q = np.array([np.cos(roll_half), np.sin(roll_half), 0.0, 0.0])
    tilt_half = np.radians(tilt_deg) / 2.0
    tilt_q = np.array([np.cos(tilt_half), 0.0, np.sin(tilt_half), 0.0])
    return quat_multiply(tilt_q, roll_q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


# Fixed quaternion equivalent of Camera.set_local_pose's own internal W_U_TRANSFORM - the
# world-axes -> USD-native-axes correction it silently right-multiplies onto the given orientation
# whenever camera_axes="world" (the default, confirmed by reading Isaac Sim's own source:
# isaacsim.sensors.camera.camera.W_U_TRANSFORM = [[0,0,-1],[-1,0,0],[0,1,0]], converted to a
# quaternion here). This constant is why camera_pan_tilt_quat below has to explicitly undo it (and
# the mount's own world rotation) before applying a genuine world-space yaw, rather than just
# adding a third raw-axis rotation into camera_head_mount_quat's composition - naively rotating
# about a raw local X, Y, or Z axis was tried in every composition order (before/after/between the
# existing roll+tilt) and reproducibly showed up as an unwanted EXTRA TILT instead of a clean pan
# in every single case, confirmed via many live render comparisons - only this explicit undo
# produces a genuinely decoupled left/right pan (confirmed live: the box shifts purely
# horizontally, with the horizon/amount-of-floor-visible unchanged, unlike every raw-axis attempt).
CAMERA_WU_QUAT = np.array([0.5, 0.5, -0.5, -0.5])


def camera_pan_tilt_quat(roll_deg: float, tilt_deg: float, pan_deg: float, mount_world_quat: np.ndarray) -> np.ndarray:
    """Local orientation for Camera.set_local_pose (relative to HEAD_CAMERA_MOUNT) that applies
    `pan_deg` as a genuine world-space yaw (rotation about the global up axis) on top of the
    existing roll+tilt correction (camera_head_mount_quat), regardless of the mount's own current
    world orientation - pass `mount_world_quat` as HEAD_CAMERA_MOUNT's current world orientation
    (read live: it moves with the torso/arm chain, even though nothing normally drives it away
    from rest during calibration). At pan_deg=0 this is mathematically guaranteed to reduce to
    exactly camera_head_mount_quat(roll_deg, tilt_deg), regardless of mount_world_quat.
    """
    q_base = camera_head_mount_quat(roll_deg, tilt_deg)
    q_cam_world = quat_multiply(quat_multiply(mount_world_quat, q_base), CAMERA_WU_QUAT)
    pan_half = np.radians(pan_deg) / 2.0
    q_yaw = np.array([np.cos(pan_half), 0.0, 0.0, np.sin(pan_half)])
    q_new_world = quat_multiply(q_yaw, q_cam_world)
    q_new_local = quat_multiply(quat_multiply(quat_conjugate(mount_world_quat), q_new_world), quat_conjugate(CAMERA_WU_QUAT))
    return q_new_local / np.linalg.norm(q_new_local)


# Real cardboard-box props from Isaac's warehouse/logistics environment set (plain generic
# shipping boxes, not branded grocery items) - see spawn_real_box/make_box_dynamic below for why
# these need extra physics authoring these ship as static (collision-only) meshes, unlike the
# earlier YCB grocery-box assets this replaced. Native footprints (x,y,z meters, measured via a
# live AABB probe at scale 1.0): CardBoxD ~(0.38, 0.25, 0.15), CardBoxC ~(0.50, 0.50, 0.25),
# CardBoxA ~(0.70, 0.50, 0.50). BOX_ASSET_MAIN is the smallest so it also fits the pushcart deck
# (0.6 x 0.45m) on --cube-start=cart sessions; CUBE2/CUBE3 are table-side only (see
# --extra-boxes) and are genuinely different assets, not the same one rescaled. The table itself
# is 0.8m x 2.8m (x,y) - CardBoxA's 0.7m width leaves only ~0.05m margin on the table's x-axis,
# worth checking visually (Stage-panel + F) rather than assuming it clears.
BOX_ASSET_MAIN = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd"
BOX_ASSET_CUBE2 = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd"
BOX_ASSET_CUBE3 = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"

# Room footprint (meters, world xy) - unrelated to the pick-place task itself, just needs to be
# large enough to contain the table/cart/robot cluster (which is now built around the origin,
# not stream_demo.py's [3.0, 1.5, 0.0]); reused as-is from stream_demo.py.
ROOM_MIN = (-6.0, -8.0)
ROOM_MAX = (12.0, 10.0)
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1

# How far (meters, along the table's near-edge normal) the robot parks from the table/cart
# cluster, and how much clearance sits between the table and cart footprints. Both are first
# guesses, not verified live - the module docstring's Stage 0 manual check is what tells you
# whether these need adjusting (robot too far to reach -> lower ROBOT_APPROACH_GAP_M; robot
# body collides with the table/cart -> raise it).
ROBOT_APPROACH_GAP_M = 0.9
CART_TABLE_GAP_M = 0.15

# Table2 (--place-target table2) - an alternative to the pushcart: a second table, same asset and
# height as the main one, placed to table1's *side* - offset along X (table1's short, 0.8m axis)
# rather than ahead along Y (table1's long, 2.8m axis), so the robot approaches table2's long
# (2.8m) edge, not its narrow 0.8m end. table2_side_sign picks which side (+1 = table1's +X/xmax
# side, -1 = -X/xmin side, same side the robot parks on) - flip it if the layout reads backward
# once viewed live; nothing else depends on the sign. TABLE2_GAP_M can now be a real, walkable gap
# (not the old near-zero clearance) because chassis_forward (see state_names) means the robot
# actually driving there is now part of what the policy learns, not something to avoid - unlike
# the old design, which kept this tight specifically so a single parked pose could reach table2 by
# arm swing alone, since driving used to be invisible to the recorder. TABLE2_EDGE_INSET_M plays
# the same role as before (the full 0.8 x 2.8m table's own centroid is still far outside reach),
# just measured from table2's near X-facing edge now instead of its near Y-facing edge. Both are
# first guesses, not verified live - same Stage 0 caveat as ROBOT_APPROACH_GAP_M above.
TABLE2_GAP_M = 1.2
TABLE2_EDGE_INSET_M = 0.3
TABLE2_SIDE_SIGN = 1.0

# Gap (meters, edge to edge) between adjacent boxes when --extra-boxes lays out 3 side by side on
# the table - kept generous so the boxes are clearly separate pick-up targets, not crowded
# together. Confirmed live against the actual table asset's footprint (0.8m x 2.8m, x by y): at
# this gap, cube2/cube3 sit at y=+-0.875 with ~0.28m clearance to the table's y-edge (table's
# y-half-extent is 1.4m) - still comfortably on the table. If box sizes are changed via
# --cube2-scale/--cube3-scale, re-check via the startup geometry diagnostic printout rather than
# assuming this still fits.
CUBE_ROW_GAP_M = 0.4

DRIVE_KEY_AXES = {
    carb.input.KeyboardInput.W: (0, 1.0),
    carb.input.KeyboardInput.S: (0, -1.0),
    carb.input.KeyboardInput.A: (1, 1.0),
    carb.input.KeyboardInput.D: (1, -1.0),
    carb.input.KeyboardInput.Q: (2, 1.0),
    carb.input.KeyboardInput.E: (2, -1.0),
}


def compute_drive_command(held_keys: set, drive_speed: float, turn_speed: float) -> list:
    scale = (drive_speed, drive_speed, turn_speed)
    command = [0.0, 0.0, 0.0]
    for key in held_keys:
        if key in DRIVE_KEY_AXES:
            axis, sign = DRIVE_KEY_AXES[key]
            command[axis] += sign * scale[axis]
    return command


def robot_heading_yaw(orientation_wxyz: np.ndarray) -> float:
    """Yaw (rotation about Z, radians) of the articulation root's local +X axis in world space -
    ported as-is from stream_demo.py (same Galbot G1 asset/root prim). This is NOT the direction
    the robot actually drives - see ROBOT_FORWARD_OFFSET_RAD."""
    w, x, y, z = orientation_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# Same -pi/2 offset as stream_demo.py's ROBOT_FORWARD_OFFSET_RAD (confirmed there empirically:
# robot_heading_yaw() reads 90deg ahead of the direction command=[1,0,0] actually drives) - same
# asset/root prim, so it applies here too. Only needed here for the chassis_forward state dim (see
# state_names below); nothing else in this file previously needed heading.
ROBOT_FORWARD_OFFSET_RAD = np.pi / 2.0


def robot_forward_reference(robot: SingleArticulation) -> tuple:
    """(position_xy, forward_unit_vector) for the chassis_forward state dim - call once at the
    start of a recording attempt (B-press or R-reset) to capture the reference frame that
    forward_displacement_m gets measured against for the rest of that attempt."""
    position, orientation_wxyz = robot.get_world_pose()
    heading = robot_heading_yaw(orientation_wxyz) - ROBOT_FORWARD_OFFSET_RAD
    return np.array(position[:2]), np.array([np.cos(heading), np.sin(heading)])


def forward_displacement(robot: SingleArticulation, start_pos_xy: np.ndarray, forward_dir: np.ndarray) -> float:
    """Signed distance (meters) the chassis has moved along forward_dir since start_pos_xy -
    see state_names' chassis_forward note above."""
    position, _ = robot.get_world_pose()
    return float(np.dot(np.array(position[:2]) - start_pos_xy, forward_dir))


# Arm/hand/torso jog controls - ported as-is from stream_demo.py / ../Robot_project/
# capture_cube_rgbd.py (same asset, same joints); see stream_demo.py's module docstring and the
# comments above each of these constants there for the full derivation.
ARM_FORWARD_POSE = {
    "left": [0.0, -1.308997, 0.0, 0.0, 0.0, 0.0, 0.0],
    "right": [0.0, -1.608100, 0.0, 0.0, 0.0, 0.0, 0.0],
}
ARM_OPEN_POSE = {
    "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
ARM_SWING_KEYS = {
    carb.input.KeyboardInput.O: -1.0,  # toward ARM_OPEN_POSE
    carb.input.KeyboardInput.U: 1.0,  # toward ARM_FORWARD_POSE
}

ARM_HAND_UPDOWN_JOINT_INDEX = 3  # joint4, 0-indexed into the 7-joint [joint1..joint7] chain
ARM_HAND_DOWN_MAX_RAD = 0.6
ARM_HAND_UP_MAX_RAD = 2.5
HAND_UPDOWN_KEYS = {
    carb.input.KeyboardInput.L: -1.0,  # toward hand down
    carb.input.KeyboardInput.J: 1.0,  # toward hand up
}

# 97.5747deg is this drive joint's own authored hard limit (both hands) - but live testing
# (pinching in open air, nothing between the fingers) showed the console diagnostic climbing
# perfectly smoothly the entire way, then simply stopping dead the instant it reached exactly this
# value, with the robot visibly breaking (arm-flinging) at that same moment - strongly suggesting
# the two jaws' collision meshes aren't clearanced to actually reach the joint's full authored
# limit without overlapping each other once nothing (no real object) stops them early, and a
# strongly-driven (600000 stiffness - see dump_arm_joint_limits) joint trying to push two
# overlapping rigid bodies further into each other is exactly the kind of sudden large contact
# force this project has repeatedly found causes the fling instability. GRIPPER_CLOSE_MAX_RAD_
# RAW keeps the true joint limit on record; the 10% margin below is what's actually commanded, so
# the fingers are never driven to their absolute mechanical end-stop with nothing between them.
# Real grasped objects (the box, the cart handle) are far thicker than this margin and would stop
# the fingers well before either limit is ever approached, so this shouldn't cost any real grip
# capability. UNVERIFIED LIVE (no Isaac Sim available while writing this) - re-run the exact same
# "pinch in open air" test and confirm it no longer breaks at full closure; tighten the margin
# further (lower the 0.9 factor) if it still breaks, since that would mean the true overlap point
# is even earlier than a 10% margin accounts for.
GRIPPER_CLOSE_MAX_RAD_RAW = np.radians(97.57470703125)
GRIPPER_CLOSE_MAX_RAD = 0.9 * GRIPPER_CLOSE_MAX_RAD_RAW
GRIPPER_SPEED_RAD_S = 2.5
GRIPPER_KEYS = {
    carb.input.KeyboardInput.N: -1.0,  # toward open
    carb.input.KeyboardInput.M: 1.0,  # toward closed
}
# This is the actual bottleneck on how fast M/N open/close feel, not GRIPPER_SPEED_RAD_S above -
# the joint can only track a moving target at roughly GRIPPER_MAX_LEAD_RAD/physics_dt. Was
# temporarily raised to 0.02 (~2.5x faster) per a speed request, then REVERTED back to 0.008 -
# live testing with 0.02 active reproduced the actual "hand breaks" failure: gripping the pushcart
# handle and holding M made the fingers visibly judder (confirmed in the console too - both
# gripper joints repeatedly hit an exact +-0.5 rad/s reading, a stick-slip pattern against contact
# resistance), and that sustained juddering is what then destabilized the whole arm into a fling,
# visibly breaking the fingers. A looser lead directly means the drive pushes harder/faster into
# resistance each step, which is very likely what fed this oscillation - speed and stability were
# in tension here, and this reverts back to the value already known to hold up under contact
# (confirmed across multiple earlier tests with the same "grip the handle" scenario before the
# lead was loosened). Do not raise this again without a live test of gripping something rigid and
# holding it for several seconds, not just an open-air open/close check.
GRIPPER_MAX_LEAD_RAD = 0.008

# Live instability watchdog for the "hand keeps breaking when I pinch the handle and move" report -
# no arm/gripper joint should ever move this fast during normal teleop (every jog control here is
# lead-clamped to small per-step corrections), so a joint velocity above this is treated as a real
# instability spike (the arm-flinging failure mode documented throughout this file), not ordinary
# motion. 3.0 rad/s is a conservative guess, not measured against an actual observed fling event -
# lower it if it doesn't trigger on a fling you can see happening, or raise it if normal fast
# teleop motion false-triggers it.
GRIPPER_WATCHDOG_VELOCITY_RAD_S = 3.0

# Effort-based compliance: stop advancing a joint's target further once it's under real sustained
# load, instead of continuing to command it forward regardless of resistance - this is a genuine
# control-scheme gap, not an IK problem (this file never uses IK; every joint here is a fixed,
# hand-authored position target, none of them aware of contact force). clamp_to_actual already
# bounds how far the target can outrun the actual position each step, which prevents instant force
# spikes, but does nothing to stop a joint being slowly dragged off-target by a sustained push -
# exactly the mechanism behind every fling traced in this file's history (e.g. left_arm_joint5
# recorded at -66.2Nm/-75.1Nm during two separate pinch-and-push tests, vs. under ~0.2Nm during
# all-clear free motion in every clean log). 15.0 sits well above that free-motion noise floor but
# below the observed danger-zone efforts - a reasoned choice, not measured against a controlled
# sweep. Only gates the arm's actively-driven joints (swing/joint2, hand-updown/joint4, wrist-
# rotate/joint5) - the gripper's own drive joint's effort stayed under 0.01Nm even while visibly
# juddering in every log, so effort isn't a useful signal there; that judder looks to be a
# velocity-limit phenomenon, not a force one, and needs a different fix if it turns out to matter
# on its own.
JOINT_EFFORT_LIMIT_NM = 15.0

# How often to print the gripper-specific diagnostic (target vs. actual vs. velocity vs. effort)
# while M/N is held or the gripper is off its target - see the print site below, added
# specifically to trace "the hand breaks only when I pinch (grab) the pushcart handle": rather
# than only reacting to a full watchdog-level spike, this gives a continuous trace of what the
# gripper joint itself is doing throughout a grab, in case a problem shows up building up
# gradually (e.g. tracking error growing) before ever crossing the spike threshold above.
GRIPPER_DIAG_PERIOD_S = 0.2

# Pushing the pushcart by its handle needs the pinch grip to transmit real push force without
# slipping - the reported failure ("robot moves back, cart doesn't move, hand loses the grab") is
# a friction problem, not a closing-force problem: GRIPPER_MAX_LEAD_RAD above is deliberately kept
# this tight specifically because closing harder against something rigid has already been
# observed to fling/break the arm (see its own comment) - so getting more holding force by closing
# harder isn't a safe option, closing harder is exactly the thing already ruled out. Raising the
# fingertip surfaces' own friction coefficient instead lets the SAME capped closing force hold
# better before slipping, without touching the stability-critical lead clamp at all. Matches the
# "next lever" CLAUDE.md already flagged for the box hug's own friction-only hold, applied here to
# the gripper-handle contact instead. 1.2 is a rubber-like grip guess (typical rubber-on-metal
# static friction is roughly 1.0-1.5) - not measured against this specific asset's fingertip mesh.
GRIPPER_FRICTION_COEFF = 1.2

# C/V jog joint5, ,/. jog joint6 - two independent wrist reorientation controls, re-enabled
# together per request to experiment with a different grab approach. joint5 (C/V) is confirmed
# live by the user as the correct rotation for the flat/horizontal gripper fix; joint6 (,/.) is
# the other live-tested candidate from that same X/Y/Z investigation (see git history for the
# full derivation and the two wrong guesses on joint7 before this per-axis setup existed).
# joint5's own axis lands on link7's local NEGATIVE X; joint6's lands on link7's local Y (both
# derived by composing joint5/6/7's origin rpy chain from the URDF, not guessed) - i.e. C/V = X,
# ,/. = Y (easy to misremember since it's the opposite of the more common X-then-Y reading order).
# Hard limits are each joint's own from the URDF, not estimated. Sign is applied identically to
# both arms for both - confirmed by composing both arms' axis-sign differences for Y (joint6's
# axis and joint7's mount sign both flip between arms and cancel); only an assumption for X, not
# derived - flip a given axis's KEYS dict sign for one arm specifically if only one hand rotates
# backward.
WRIST_X_JOINT_INDEX = 4  # joint5, 0-indexed into the 7-joint [joint1..joint7] chain
WRIST_X_MIN_RAD = -2.91697
WRIST_X_MAX_RAD = 2.91697
WRIST_X_KEYS = {
    carb.input.KeyboardInput.C: -1.0,
    carb.input.KeyboardInput.V: 1.0,
}

WRIST_Y_JOINT_INDEX = 5  # joint6
WRIST_Y_MIN_RAD = -0.7354
WRIST_Y_MAX_RAD = 0.8227
WRIST_Y_KEYS = {
    carb.input.KeyboardInput.COMMA: -1.0,
    carb.input.KeyboardInput.PERIOD: 1.0,
}

WRIST_Z_JOINT_INDEX = 6  # joint7 - its own <axis> IS link7's local Z, exactly, by definition
WRIST_Z_MIN_RAD = -1.5382
WRIST_Z_MAX_RAD = 1.5382
WRIST_Z_KEYS = {
    carb.input.KeyboardInput.LEFT_BRACKET: -1.0,
    carb.input.KeyboardInput.RIGHT_BRACKET: 1.0,
}

TORSO_UP_POSE = [0.0, 0.0, 0.0, 0.0, 0.0]
TORSO_DOWN_POSE = [0.8, 2.3, 1.55, 0.0, 0.0]
TORSO_HEIGHT_KEYS = {
    carb.input.KeyboardInput.I: -1.0,  # toward TORSO_UP_POSE
    carb.input.KeyboardInput.K: 1.0,  # toward TORSO_DOWN_POSE
}

# Camera pan/tilt calibration keys (arrow keys - unused everywhere else in this file). Signs
# confirmed live by rendering, not assumed: increasing CAMERA_PAN_DEG rotates the view to look
# further LEFT (a centered object shifts toward the right edge of frame), increasing
# CAMERA_TILT_DEG pitches further DOWN (a centered object shifts toward the bottom of frame, the
# horizon rises) - see camera_pan_tilt_quat/camera_head_mount_quat.
CAMERA_ROTATE_KEYS_PAN = {
    carb.input.KeyboardInput.LEFT: 1.0,
    carb.input.KeyboardInput.RIGHT: -1.0,
}
CAMERA_ROTATE_KEYS_TILT = {
    carb.input.KeyboardInput.DOWN: 1.0,
    carb.input.KeyboardInput.UP: -1.0,
}

# Starting arm/hand pose on launch and every reset - matches a specific teleoperated pose the
# user confirmed live via a recorded episode (episode_0010, 2026-08-28: left/right arm_joint2
# ~-1.067rad, arm_joint4 ~+1.173rad on both arms), not the fully-open rest pose. Expressed as
# swing-fraction/hand-updown values (not raw joint angles) since that's what the jog loop below
# actually drives - left/right fractions differ despite the near-identical recorded joint2 angle
# because ARM_FORWARD_POSE's joint2 target differs per arm (asymmetric shoulder mount), while
# arm_swing_rate normalizes both arms to the same physical rad/s, not fraction/s. The transition
# from the USD asset's authored rest pose (0 rad) to this pose happens gradually (confirmed live:
# smoothly converges over ~5s in free space, no instability) via the existing clamp_to_actual
# mechanism already used for jogging - no direct joint teleport. The per-step lead cap
# (ARM_CONTACT_MAX_LEAD_RAD) is not actually the limiting factor for this unobstructed motion -
# confirmed live the joints track ~6x slower than the cap allows (presumably the drive's own
# tracking bandwidth), so full convergence takes several seconds, not the single physics step the
# lead cap alone would suggest.
STARTING_LEFT_ARM_SWING_FRACTION = 0.815
STARTING_RIGHT_ARM_SWING_FRACTION = 0.663
STARTING_HAND_UPDOWN_RAD = 1.173

# Default wrist-rotate values (see WRIST_X/Y/Z_JOINT_INDEX above), applied at launch/reset.
# X (joint5) is live-calibrated via C/V and the console's "[wrist] joint5=...deg" readout -
# -0.7400 rad (-42.4deg) is where the gripper actually reads flat/horizontal against the
# reference photo, confirmed live, not guessed. Y (joint6) and Z (joint7) have no live-calibrated
# value yet - left at 0/no-offset; hold , / . or [ / ] and watch the matching "[wrist] jointN=...
# deg" readout the same way if a non-zero default turns out to help the new grab approach being
# tried.
STARTING_WRIST_X_RAD = -0.7400
STARTING_WRIST_Y_RAD = 0.0
STARTING_WRIST_Z_RAD = 0.0

MAX_JOINT_LEAD_RAD = 0.3
# Tighter than stream_demo.py/capture_cube_rgbd.py's 0.1 rad - live-observed here (holding U
# pressed into the bigger/heavier real box, arms flung the robot after a sustained hold, not on
# first contact) that 0.1 rad of continuously-reasserted lead is enough sustained torque against
# this task's larger contact area to destabilize the robot over time. Every physics step this
# clamp recomputes the target as "actual position +/- max_lead", so as long as a swing key is
# held into something that isn't yielding, the controller keeps trying to advance that lead
# indefinitely - there's no per-frame magnitude that's safe forever, only "tight enough that the
# sustained-contact steady-state force stays survivable." Same principle as GRIPPER_MAX_LEAD_RAD
# below, just less extreme since the arm contact area/torque budget is larger than a fingertip
# pinch. Not yet swept to find a real ceiling - lower further if a sustained hug still flings the
# robot, same as this constant's sibling-project counterpart says for its own value.
ARM_CONTACT_MAX_LEAD_RAD = 0.03


def arm_swing_rate(side: str, arm_speed: float) -> float:
    delta = abs(np.array(ARM_FORWARD_POSE[side]) - np.array(ARM_OPEN_POSE[side])).max()
    return arm_speed / float(delta)


def clamp_to_actual(target: np.ndarray, actual: np.ndarray, max_lead: float = MAX_JOINT_LEAD_RAD) -> np.ndarray:
    return np.clip(target, actual - max_lead, actual + max_lead)


def arm_dof_indices(robot: SingleArticulation, side: str) -> list:
    return [robot.get_dof_index(f"{side}_arm_joint{i}") for i in range(1, 8)]


def leg_dof_indices(robot: SingleArticulation) -> list:
    return [robot.get_dof_index(f"leg_joint{i}") for i in range(1, 6)]


def gripper_dof_indices(robot: SingleArticulation, side: str) -> list:
    return [robot.get_dof_index(f"{side}_gripper_joint")]


def read_wheel_geometry(holonomic_graph: str) -> dict:
    setup_node = f"{holonomic_graph}/usd_setup_holonomic_robot"
    return {
        "wheel_radius": og.Controller.attribute(f"{setup_node}.outputs:wheelRadius").get(),
        "wheel_positions": og.Controller.attribute(f"{setup_node}.outputs:wheelPositions").get(),
        "wheel_orientations": og.Controller.attribute(f"{setup_node}.outputs:wheelOrientations").get(),
        "mecanum_angles": og.Controller.attribute(f"{setup_node}.outputs:mecanumAngles").get(),
        "wheel_axis": og.Controller.attribute(f"{setup_node}.outputs:wheelAxis").get(),
        "up_axis": og.Controller.attribute(f"{setup_node}.outputs:upAxis").get(),
        "wheel_dof_names": list(og.Controller.attribute(f"{setup_node}.outputs:wheelDofNames").get()),
    }


def remove_ros2_control_graphs(stage) -> None:
    stage.RemovePrim(Sdf.Path(f"{ROBOT_PRIM}/OmniChassis/Graph/holonomic_controller"))
    stage.RemovePrim(Sdf.Path(f"{ROBOT_PRIM}/OmniChassis/Graph/ROS_Odometry"))
    stage.RemovePrim(Sdf.Path(f"{ROBOT_PRIM}/Graph/ROS_JointStates"))


def build_drive_controller(geometry: dict) -> HolonomicController:
    return HolonomicController(
        name="galbot_drive",
        wheel_radius=np.asarray(geometry["wheel_radius"]),
        wheel_positions=np.asarray(geometry["wheel_positions"]),
        wheel_orientations=np.asarray(geometry["wheel_orientations"]),
        mecanum_angles=np.asarray(geometry["mecanum_angles"]),
        wheel_axis=np.asarray(geometry["wheel_axis"]),
        up_axis=np.asarray(geometry["up_axis"]),
        max_linear_speed=3.0,
        max_angular_speed=3.0,
        max_wheel_speed=30.0,
        linear_gain=-1.0,
    )


def build_room(stage) -> None:
    (x0, y0), (x1, y1) = ROOM_MIN, ROOM_MAX
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width, depth = x1 - x0, y1 - y0

    walls = [
        (cx, y0, width / 2.0 + WALL_THICKNESS, WALL_THICKNESS),
        (cx, y1, width / 2.0 + WALL_THICKNESS, WALL_THICKNESS),
        (x0, cy, WALL_THICKNESS, depth / 2.0),
        (x1, cy, WALL_THICKNESS, depth / 2.0),
    ]
    for i, (wx, wy, hx, hy) in enumerate(walls):
        wall = UsdGeom.Cube.Define(stage, f"/World/Wall{i}")
        wall.AddTranslateOp().Set(Gf.Vec3d(wx, wy, WALL_HEIGHT / 2.0))
        wall.AddScaleOp().Set(Gf.Vec3f(hx, hy, WALL_HEIGHT / 2.0))
        wall.CreateDisplayColorAttr([(0.75, 0.73, 0.68)])
        UsdPhysics.CollisionAPI.Apply(wall.GetPrim())


def compute_world_aabb(bbox_cache, prim_path: str) -> np.ndarray:
    prim = get_prim_at_path(prim_path)
    r = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array([*r.GetMin(), *r.GetMax()])


def sample_pose_jitter(rng: np.random.Generator, jitter_m: float, yaw_jitter_deg: float) -> tuple:
    """Sample a random (dx, dy, yaw_deg, yaw_quat) offset for a spawn/reset pose - used for both
    the main pick box (--box-jitter-m/--box-yaw-jitter-deg) and table2
    (--table2-jitter-m/--table2-yaw-jitter-deg). dx/dy are uniform over a disk (not a square) of
    radius jitter_m, so every direction is equally likely rather than corners being favored. yaw
    is uniform in [-yaw_jitter_deg, +yaw_jitter_deg] and returned both as degrees (for logging) and
    as a quaternion in the same (w, x, y, z) / rotation-about-Z convention as quat_multiply/
    camera_pan_tilt_quat above (safe to use directly as an orientation since both objects' un-
    jittered pose is axis-aligned identity orientation)."""
    r = jitter_m * np.sqrt(rng.uniform(0.0, 1.0))
    theta = rng.uniform(0.0, 2.0 * np.pi)
    dx, dy = r * np.cos(theta), r * np.sin(theta)
    yaw_deg = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
    yaw_half = np.radians(yaw_deg) / 2.0
    yaw_quat = np.array([np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)])
    return dx, dy, yaw_deg, yaw_quat


def sample_cube_scale(rng: np.random.Generator, args, cycle_index: int = 0) -> float:
    """The main box's scale for one spawn/reset - fixed at --cube-scale unless --cube-scale-min/
    --cube-scale-max are both set (validated together at argparse time). With --cube-scale-cycle,
    steps deterministically through small (min) -> medium (midpoint) -> big (max) -> repeat, keyed
    off cycle_index (the caller's spawn/reset counter, not randomized); otherwise drawn uniformly
    at random from [min, max]. See --cube-scale-min's help for why this exists and its
    live-verification caveat."""
    if args.cube_scale_min is None:
        return args.cube_scale
    if args.cube_scale_cycle:
        sizes = (args.cube_scale_min, (args.cube_scale_min + args.cube_scale_max) / 2.0, args.cube_scale_max)
        return sizes[cycle_index % 3]
    return float(rng.uniform(args.cube_scale_min, args.cube_scale_max))


def place_on_ground(bbox_cache, prim_path: str, x: float, y: float, scale: float = 1.0, z_scale: float = None) -> np.ndarray:
    """Move a freshly-referenced (identity-transform) prim so its footprint is centered at
    (x, y) and its lowest point rests on z=0. Ported from
    ../Robot_project/capture_cube_rgbd.py - needed here (unlike stream_demo.py, which hardcodes
    the table's position) so the table/cart/robot cluster's exact AABBs are known and
    reproducible, which the cart-adjacency placement below depends on.

    `z_scale` defaults to `scale` (uniform scaling, the original behavior) but can be passed
    separately to scale only the height - e.g. TABLE_HEIGHT_SCALE, which shortens the table's
    legs without shrinking its tabletop footprint (used for box/cart placement elsewhere).
    """
    if z_scale is None:
        z_scale = scale
    aabb0 = compute_world_aabb(bbox_cache, prim_path) * np.array([scale, scale, z_scale, scale, scale, z_scale])
    center_x0 = (aabb0[0] + aabb0[3]) / 2.0
    center_y0 = (aabb0[1] + aabb0[4]) / 2.0
    position = np.array([x - center_x0, y - center_y0, -aabb0[2]])
    SingleXFormPrim(prim_path, position=position, scale=np.array([scale, scale, z_scale]))
    bbox_cache.Clear()
    return compute_world_aabb(bbox_cache, prim_path)


def place_on_surface(bbox_cache, prim_path: str, x: float, y: float, surface_z: float, scale: float = 1.0) -> np.ndarray:
    """Like place_on_ground, but rests the prim's lowest point on `surface_z` (e.g. a tabletop or
    cart deck) instead of the floor. Same precondition as place_on_ground: `prim_path` must still
    be at its just-referenced identity transform when this is called (the scale-then-measure
    trick - multiplying the identity-transform AABB by `scale` - only gives the right answer
    before any transform has been authored on the prim)."""
    aabb0 = compute_world_aabb(bbox_cache, prim_path) * scale
    center_x0 = (aabb0[0] + aabb0[3]) / 2.0
    center_y0 = (aabb0[1] + aabb0[4]) / 2.0
    position = np.array([x - center_x0, y - center_y0, surface_z - aabb0[2]])
    SingleXFormPrim(prim_path, position=position, scale=np.array([scale, scale, scale]))
    bbox_cache.Clear()
    return compute_world_aabb(bbox_cache, prim_path)


def scaled_footprint(bbox_cache, prim_path: str, scale: float) -> np.ndarray:
    """Non-mutating: what place_on_surface/place_on_ground would measure at `scale`, without
    moving the prim. Used to size a box before deciding where to place it (see the table row
    layout in main()) - must also be called before any transform has been authored on the prim,
    same precondition as place_on_surface.
    """
    return compute_world_aabb(bbox_cache, prim_path) * scale


def make_box_dynamic(prim_path: str, mass: float) -> None:
    """Author RigidBodyAPI + MassAPI on a referenced box prop's root, and override its mesh
    child's collision approximation to convexHull. The warehouse cardboard-box props this is used
    for (BOX_ASSET_MAIN/CUBE2/CUBE3) ship as static, collision-only meshes - confirmed live: the
    mesh's collision approximation defaults to "none" (an exact triangle mesh), which PhysX
    accepts for a static collider but rejects for a *dynamic* rigid body - only convex shapes are
    valid there. convexHull is a safe choice for a box-shaped mesh (tested live: a controlled
    two-plate squeeze against it settled cleanly, no NaN/instability, and convexHull vs
    boundingCube made no meaningful difference in that test - the shape choice itself does not
    appear to be what destabilizes a *sustained* two-arm hug, see BOX_CONTACT_OFFSET_M's comment
    below for the more likely cause).

    Also authors the same PhysxCollisionAPI contact tuning `isaacsim.core.api.objects.DynamicCuboid`
    gives every cuboid by default (rest_offset=0.0, contact_offset=0.1m, torsional_patch_radius=1.0,
    min_torsional_patch_radius=0.8) - the earlier procedural-cube version of this scene got this
    for free; these real mesh assets don't, and torsional patch radius specifically matters for a
    friction-only hug (it's what resists the box twisting/slipping in the grip). `Apply()` is a
    no-op if an API is already present, so this is safe to call even on an asset that already had
    physics authored.
    """
    prim = get_prim_at_path(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    mesh_prim = next((c for c in prim.GetAllChildren() if c.GetTypeName() == "Mesh"), None)
    if mesh_prim is not None:
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("convexHull")
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
        physx_collision.CreateRestOffsetAttr().Set(0.0)
        physx_collision.CreateContactOffsetAttr().Set(0.1)
        physx_collision.CreateTorsionalPatchRadiusAttr().Set(1.0)
        physx_collision.CreateMinTorsionalPatchRadiusAttr().Set(0.8)


def spawn_real_box(
    bbox_cache, assets_root_path: str, usd_relpath: str, prim_path: str, x: float, y: float, surface_z: float, scale: float, mass: float
) -> np.ndarray:
    """Reference a real box asset (see BOX_ASSET_MAIN/CUBE2/CUBE3), make it a dynamic rigid body
    (see make_box_dynamic), and place it resting on `surface_z`. Friction is left at the asset's
    own baked-in default for now - not yet worth the risk of mixing the isaacsim.core.api (legacy)
    and isaacsim.core.experimental physics-material APIs without live verification; if the hug
    hold proves unreliable, binding a custom high-friction PhysicsMaterial here is the next thing
    to try, not a kinematic attach (see the module docstring).
    """
    add_reference_to_stage(usd_path=assets_root_path + usd_relpath, prim_path=prim_path)
    make_box_dynamic(prim_path, mass)
    return place_on_surface(bbox_cache, prim_path, x=x, y=y, surface_z=surface_z, scale=scale)


# Pushcart geometry - ported from ../Robot_project/capture_cube_rgbd.py's build_pushcart, with
# one change: a deck_riser_height parameter (see pushcart_deck_top_z / --deck-riser) inserted
# between the caster assembly and the deck, since the stock ~0.15m deck height was designed for
# "push by the handle," not "place a box here," and is likely well below table height.
#
# Sized up from the original (0.45, 0.225)/0.05 wheel radius/4.4kg per request: bigger deck
# footprint, taller undercarriage (bigger wheel radius raises deck_bottom_z - see
# pushcart_deck_top_z). UNVERIFIED LIVE - re-run the Stage 0 reach/hug cycle and re-check
# --deck-riser/ROBOT_APPROACH_GAP_M/CART_TABLE_GAP_M against the new taller/bigger geometry before
# trusting it for real collection.
PUSHCART_DECK_HALF_EXTENT = (0.55, 0.30)  # was (0.45, 0.225) - bigger deck footprint
PUSHCART_DECK_THICKNESS = 0.04  # was 0.03
PUSHCART_WHEEL_RADIUS = 0.08  # was 0.05 - raises deck_bottom_z, making the whole cart taller
PUSHCART_HANDLE_POST_HEIGHT = 0.85  # was 0.75

# Mass and rolling friction were originally both raised together (4.4kg->25kg chassis, intended to
# resist an accidental touch) but that conflated two different things: chassis MASS also
# determines how much reaction force gets transmitted back through a gripped handle into the
# robot's wrist during a deliberate push, and 25kg turned out to be enough to overpower the
# wrist's safety-limited holding force (ARM_CONTACT_MAX_LEAD_RAD) under real load - live-observed
# as arm_joint5 creeping/oscillating well off its held target while pushing the cart (see git
# history / session notes for the watchdog log that showed this). Splitting the two concerns
# instead: chassis mass brought back down close to the original (inertia/reaction-force budget
# stays small enough for the wrist to hold against), while CASTER_ROLLING_FRICTION_NM is raised
# steeply instead (resists a light accidental bump via static friction at the wheels themselves,
# independent of mass - a real push should still have enough sustained force to overcome it).
# UNVERIFIED LIVE - re-test both "accidentally bump it while parking" and "deliberately grip and
# push it" before trusting this balance; raise CASTER_ROLLING_FRICTION_NM further if light bumps
# still move it, or lower PUSHCART_CHASSIS_MASS further if the wrist still struggles under a
# deliberate push.
PUSHCART_CHASSIS_MASS = 6.0  # was 25.0 (originally 4.4) - see comment above
PUSHCART_FORK_MASS = 0.08  # was 0.05
PUSHCART_WHEEL_MASS = 0.2  # was 0.1 - bigger sphere wheels (see Wheel{i} below)
CASTER_ROLLING_FRICTION_NM = 2.0  # was 0.05 - see comment above

# Which side of the deck (in the cart's own local +/-X) the handle sits on - the caster/deck
# layout is otherwise symmetric under a 180deg yaw about Z, so "rotate the cart to the opposite
# side" is just this sign flip rather than an actual authored rotation. +1 = handle on the +X
# side (this project's new default, per request - "opposite side" from the original -1); -1 = the
# original side, ported as-is from capture_cube_rgbd.py. Flip back to -1 if the new side reads
# wrong once viewed live (e.g. the handle ends up on the side facing the robot's approach instead
# of away from it).
PUSHCART_HANDLE_SIDE_SIGN = 1.0


def pushcart_deck_top_z(deck_riser_height: float) -> float:
    """World Z of the pushcart deck's top surface for a given deck_riser_height - single source
    of truth shared between build_pushcart (which authors the deck at this height) and the
    startup geometry diagnostic in main() (which prints it against table_top_z)."""
    deck_bottom_z = 2.0 * PUSHCART_WHEEL_RADIUS + 0.02 + deck_riser_height
    return deck_bottom_z + PUSHCART_DECK_THICKNESS


def build_pushcart(stage, prim_path: str, x: float, y: float, deck_riser_height: float = 0.0) -> None:
    """Author a pushcart directly with UsdGeom/UsdPhysics primitives - see
    ../Robot_project/capture_cube_rgbd.py's build_pushcart docstring for the full derivation
    (why 9 rigid bodies, why free-swiveling casters carry Coulomb friction instead of being
    frictionless or welded, etc.). Differs from that version in deck_riser_height (raises the deck
    above the stock ~0.15m height, see pushcart_deck_top_z), PUSHCART_HANDLE_SIDE_SIGN (which side
    of the deck the handle sits on), and sized-up/heavier constants - see the constants block
    above for what changed and why. Caster fork/wheel joint geometry itself is untouched.
    """
    dx, dy = PUSHCART_DECK_HALF_EXTENT
    deck_top_z = pushcart_deck_top_z(deck_riser_height)
    deck_center_z = deck_top_z - PUSHCART_DECK_THICKNESS / 2.0
    frame_color = [(0.55, 0.55, 0.58)]

    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(PUSHCART_CHASSIS_MASS)

    deck = UsdGeom.Cube.Define(stage, f"{prim_path}/Deck")
    deck.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, deck_center_z))
    deck.AddScaleOp().Set(Gf.Vec3f(dx, dy, PUSHCART_DECK_THICKNESS / 2.0))
    deck.CreateDisplayColorAttr(frame_color)
    UsdPhysics.CollisionAPI.Apply(deck.GetPrim())

    for i, (wx, wy) in enumerate((sx * (dx - PUSHCART_WHEEL_RADIUS), sy * (dy - PUSHCART_WHEEL_RADIUS)) for sx in (-1, 1) for sy in (-1, 1)):
        wheel_center = Gf.Vec3d(wx, wy, PUSHCART_WHEEL_RADIUS)

        fork = UsdGeom.Xform.Define(stage, f"{prim_path}/CasterFork{i}")
        fork.AddTranslateOp().Set(wheel_center)
        UsdPhysics.RigidBodyAPI.Apply(fork.GetPrim())
        UsdPhysics.MassAPI.Apply(fork.GetPrim()).CreateMassAttr(PUSHCART_FORK_MASS)

        swivel = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/CasterSwivel{i}")
        swivel.CreateBody0Rel().SetTargets([Sdf.Path(prim_path)])
        swivel.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/CasterFork{i}")])
        swivel.CreateAxisAttr("Z")
        swivel.CreateLocalPos0Attr().Set(Gf.Vec3f(wheel_center))
        swivel.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        PhysxSchema.PhysxJointAPI.Apply(swivel.GetPrim()).CreateJointFrictionAttr(CASTER_ROLLING_FRICTION_NM)

        # Ball-caster-style sphere wheel (was a flat Cylinder) - a sphere has no "long axis" to
        # look wrong as it rotates, so it reads correctly rolling in any direction regardless of
        # which way CasterSwivel{i} has the fork pointed, unlike a cylinder which only looks right
        # spinning about its own Y. UsdPhysics.CollisionAPI on a UsdGeom.Sphere uses an exact
        # sphere collision approximation (no convexHull override needed, unlike the warehouse box
        # props elsewhere in this file).
        wheel = UsdGeom.Sphere.Define(stage, f"{prim_path}/Wheel{i}")
        wheel.CreateRadiusAttr(PUSHCART_WHEEL_RADIUS)
        wheel.AddTranslateOp().Set(wheel_center)
        wheel.CreateDisplayColorAttr([(0.05, 0.05, 0.05)])
        UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
        UsdPhysics.MassAPI.Apply(wheel.GetPrim()).CreateMassAttr(PUSHCART_WHEEL_MASS)

        # CasterSwivel{i} above (Z axis) and CasterSpin{i} here (Y axis) are both authored with no
        # lower/upper limit attrs, which USD Physics treats as unlimited - i.e. both the fork's
        # steering angle and the wheel's rolling spin already sweep the full 360deg, free-swiveling
        # like a real caster. Nothing to change there; the sphere wheel above is what was actually
        # missing for it to look right doing so from any fork heading.
        spin = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/CasterSpin{i}")
        spin.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/CasterFork{i}")])
        spin.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/Wheel{i}")])
        spin.CreateAxisAttr("Y")
        spin.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        spin.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        PhysxSchema.PhysxJointAPI.Apply(spin.GetPrim()).CreateJointFrictionAttr(CASTER_ROLLING_FRICTION_NM)

    post_x = PUSHCART_HANDLE_SIDE_SIGN * (dx - 0.02)
    post_radius = 0.015
    handle_top_z = deck_top_z + PUSHCART_HANDLE_POST_HEIGHT
    for i, py in enumerate((-dy + post_radius, dy - post_radius)):
        post = UsdGeom.Cylinder.Define(stage, f"{prim_path}/HandlePost{i}")
        post.CreateRadiusAttr(post_radius)
        post.CreateHeightAttr(handle_top_z - deck_center_z)
        post.CreateAxisAttr("Z")
        post.AddTranslateOp().Set(Gf.Vec3d(post_x, py, (handle_top_z + deck_center_z) / 2.0))
        post.CreateDisplayColorAttr(frame_color)
        UsdPhysics.CollisionAPI.Apply(post.GetPrim())

    bar = UsdGeom.Cylinder.Define(stage, f"{prim_path}/HandleBar")
    bar.CreateRadiusAttr(post_radius)
    bar.CreateHeightAttr(2.0 * (dy - post_radius))
    bar.CreateAxisAttr("Y")
    bar.AddTranslateOp().Set(Gf.Vec3d(post_x, 0.0, handle_top_z))
    bar.CreateDisplayColorAttr(frame_color)
    UsdPhysics.CollisionAPI.Apply(bar.GetPrim())


class RecorderState(Enum):
    IDLE = auto()
    RECORDING = auto()
    AWAITING_LABEL = auto()


class EpisodeRecorder:
    """Buffers one episode's RGB + depth frames and proprioception/action vectors in memory (a
    few seconds at 15Hz/640x480 is comfortably under a GB - depth is the big one, raw float32 at
    ~1.2MB/frame vs RGB's tens-of-KB compressed PNG - fine to hold in RAM for one episode, but
    worth knowing before recording a long session: this adds up on disk fast) and writes it to
    disk as raw_episodes/episode_NNNN/ on save(). Deliberately not LeRobot-shaped directly (no
    lerobot import here) - see convert_to_lerobot.py for the offline conversion step, run in a
    separate environment.
    """

    def __init__(self, out_dir: str, fps: float, state_names: list, camera_key: str, image_hw: tuple, task_name: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.state_names = state_names
        self.camera_key = camera_key
        self.image_hw = image_hw
        self.task_name = task_name
        self.episode_index = self._next_episode_index()
        self._reset_buffer()

    def _next_episode_index(self) -> int:
        existing = sorted(self.out_dir.glob("episode_*"))
        if not existing:
            return 0
        return int(existing[-1].name.split("_")[1]) + 1

    def _reset_buffer(self) -> None:
        self.frames: list = []
        self.depth_frames: list = []
        self.states: list = []
        self.actions: list = []

    def start(self) -> None:
        self._reset_buffer()

    def append(self, rgb: np.ndarray, depth: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.frames.append(np.ascontiguousarray(rgb[:, :, :3]))
        self.depth_frames.append(np.ascontiguousarray(depth, dtype=np.float32))
        self.states.append(state)
        self.actions.append(action)

    def discard(self) -> None:
        self._reset_buffer()

    def save(self, success: bool) -> None:
        if not self.frames:
            print("Recorder: nothing buffered, skipping save.")
            return
        ep_dir = self.out_dir / f"episode_{self.episode_index:04d}"
        frames_dir = ep_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i, rgb in enumerate(self.frames):
            Image.fromarray(rgb, mode="RGB").save(frames_dir / f"{i:06d}_rgb.png")
        for i, depth in enumerate(self.depth_frames):
            # Raw float32 meters, not a lossy colorized preview - matches capture_cube_rgbd.py's
            # save_depth precedent (np.save of the raw array) rather than streaming_server.py's
            # _depth_to_rgb (that's a viewer-only preview, not something to train on). May contain
            # inf for no-hit pixels - that's a legitimate reading, not an error; left as-is for
            # whatever consumes this later to handle (see check_raw_episodes.py's NaN-only check).
            np.save(frames_dir / f"{i:06d}_depth.npy", depth)

        state_arr = np.stack(self.states).astype(np.float32)
        action_arr = np.stack(self.actions).astype(np.float32)
        timestamps = (np.arange(len(self.frames), dtype=np.float32)) / self.fps
        np.savez(
            ep_dir / "data.npz",
            **{"observation.state": state_arr, "action": action_arr, "timestamp": timestamps},
        )

        manifest = {
            "episode_index": self.episode_index,
            "success": success,
            "fps": self.fps,
            "task": self.task_name,
            "state_dim": state_arr.shape[1],
            "action_dim": action_arr.shape[1],
            "state_names": self.state_names,
            "camera": {"key": self.camera_key, "width": self.image_hw[1], "height": self.image_hw[0]},
            "depth_capture": True,
            "num_frames": len(self.frames),
        }
        (ep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(
            f"[episode {self.episode_index:04d}] saved ({'success' if success else 'failure'}), "
            f"{len(self.frames)} frames, {len(self.frames) / self.fps:.1f}s -> {ep_dir}"
        )
        self.episode_index += 1
        self._reset_buffer()


def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not resolve the Isaac Sim assets root path (check network access).")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = get_current_stage()

    build_room(stage)

    # Lighting - missing entirely until now (build_room only builds walls, no lights), which is
    # why the scene was dark. Matches stream_demo.py's build_scene exactly.
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    distant_light.CreateIntensityAttr(3000.0)

    bbox_cache = bounds_utils.create_bbox_cache()

    add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table")
    table_aabb = place_on_ground(bbox_cache, "/World/Table", x=0.0, y=0.0, z_scale=args.table_height_scale)
    table_center_x = (table_aabb[0] + table_aabb[3]) / 2.0
    table_center_y = (table_aabb[1] + table_aabb[4]) / 2.0
    table_top_z = table_aabb[5]

    # rng created here (not down by the box, where it used to be) since table2's jitter below
    # needs it too, and table2 is built before the box.
    rng = np.random.default_rng(args.seed)

    # Pick/place partner - either the pushcart or a second table, chosen via --place-target (see
    # that flag's help and TABLE2_GAP_M's comment above). Only one is ever built - they're
    # alternative task variants, not simultaneous targets (a single parked robot pose can't reach
    # both the cart and a full-size table2 at once).
    table2_xform = None
    table2_anchor_x = table2_anchor_y = table2_anchor_z = None
    if args.place_target == "cart":
        # Cart placed beside the table, front edge flush with the table's own near (-x) edge, so
        # both sit at the same approach depth and only differ in y - see ROBOT_APPROACH_GAP_M's
        # comment above for why this exact layout is a first guess, not a verified one. Not
        # position-jittered yet (see --table2-jitter-m's help - table2 only, for now).
        dx, dy = PUSHCART_DECK_HALF_EXTENT
        target_x = table_aabb[0] + dx
        target_y = table_aabb[4] + CART_TABLE_GAP_M + dy
        build_pushcart(stage, "/World/PushCart", x=target_x, y=target_y, deck_riser_height=args.deck_riser)
        target_top_z = pushcart_deck_top_z(args.deck_riser)
    else:
        table2_half_dx = (table_aabb[3] - table_aabb[0]) / 2.0
        add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table2")
        table1_side_x = table_aabb[3] if TABLE2_SIDE_SIGN > 0 else table_aabb[0]
        table2_anchor_x = table1_side_x + TABLE2_SIDE_SIGN * (TABLE2_GAP_M + table2_half_dx)
        table2_anchor_y = table_center_y
        table2_dx, table2_dy, table2_yaw_deg, table2_yaw_quat = sample_pose_jitter(
            rng, args.table2_jitter_m, args.table2_yaw_jitter_deg
        )
        table2_aabb = place_on_ground(
            bbox_cache, "/World/Table2",
            x=table2_anchor_x + table2_dx, y=table2_anchor_y + table2_dy,
            z_scale=args.table_height_scale,
        )
        table2_xform = SingleXFormPrim("/World/Table2")
        table2_xform.set_world_pose(orientation=table2_yaw_quat)
        table2_anchor_z = float(table2_xform.get_world_pose()[0][2])
        print(
            f"[table2] initial spawn offset dx={table2_dx:+.3f}m dy={table2_dy:+.3f}m yaw={table2_yaw_deg:+.1f}deg "
            f"(--table2-jitter-m={args.table2_jitter_m} --table2-yaw-jitter-deg={args.table2_yaw_jitter_deg})"
        )
        target_x = table2_anchor_x + TABLE2_SIDE_SIGN * TABLE2_EDGE_INSET_M + table2_dx
        target_y = table2_anchor_y + table2_dy
        target_top_z = table2_aabb[5]

    add_reference_to_stage(usd_path=assets_root_path + ROBOT_ASSET, prim_path=ROBOT_PRIM)
    dump_arm_joint_limits(stage, ROBOT_PRIM)
    dump_gripper_hierarchy(stage, ROBOT_PRIM)
    stiffen_head_joints()
    stiffen_gripper_friction(stage, ROBOT_PRIM)
    # stiffen_wrist_joints(stage, ROBOT_PRIM) - DISABLED per live evidence, not just left
    # unverified: a pinch-the-handle-then-drive test with this active showed spikes escalating
    # within a single incident (-3.3 -> +7.2 -> -18.4 rad/s) and joint5/6 reaching multiple full
    # rotations (14.79/7.37 rad), spreading to the whole arm and the other arm - worse than the
    # isolated single-joint spikes seen before this was added, consistent with added stiffness
    # feeding a growing oscillation rather than damping one out. Re-enable only after
    # dump_arm_joint_limits' output is reviewed and a real root cause (very possibly: these joints
    # have no enforced PhysX rotation limit at all in this asset, unlike the reference URDF) is
    # understood - don't just try a different stiffness/damping number blind again.
    robot_spawn_x = table_aabb[0] - ROBOT_APPROACH_GAP_M
    robot_spawn_y = (table_center_y + target_y) / 2.0  # centered between table and the place-target
    place_on_ground(bbox_cache, ROBOT_PRIM, x=robot_spawn_x, y=robot_spawn_y)

    # Random per-episode spawn pose for the main pick box (see --box-jitter-m/--box-yaw-jitter-deg)
    # - box_center_x/y/box_surface_z are the tuned anchor (what used to be the exact, always-
    # identical spawn point); box_xform/box_anchor_z are captured here so the reset_requested
    # handler below can re-randomize the pose on every episode without re-running the
    # scale-then-measure placement trick (place_on_surface requires an identity-transform prim,
    # which "/World/Cube" no longer is after this first placement). rng was created earlier
    # (before table2), shared between both.
    box_center_x, box_center_y = (table_center_x, table_center_y) if args.cube_start == "table" else (target_x, target_y)
    box_surface_z = table_top_z if args.cube_start == "table" else target_top_z
    box_dx, box_dy, box_yaw_deg, box_yaw_quat = sample_pose_jitter(rng, args.box_jitter_m, args.box_yaw_jitter_deg)
    # Counts spawns/resets for --cube-scale-cycle's deterministic small->medium->big progression
    # (index 0 = small, at this very first spawn). Unused when --cube-scale-cycle is off.
    cube_scale_cycle_index = 0
    box_scale = sample_cube_scale(rng, args, cube_scale_cycle_index)
    main_box_aabb = spawn_real_box(
        bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
        x=box_center_x + box_dx, y=box_center_y + box_dy, surface_z=box_surface_z, scale=box_scale, mass=args.cube_mass,
    )
    box_xform = SingleXFormPrim("/World/Cube")
    box_xform.set_world_pose(orientation=box_yaw_quat)
    box_anchor_z = float(box_xform.get_world_pose()[0][2])
    print(
        f"[box] initial spawn offset dx={box_dx:+.3f}m dy={box_dy:+.3f}m yaw={box_yaw_deg:+.1f}deg scale={box_scale:.3f} "
        f"(--box-jitter-m={args.box_jitter_m} --box-yaw-jitter-deg={args.box_yaw_jitter_deg})"
    )

    # Two extra, bigger boxes - table side only (see CUBE_ROW_GAP_M's comment: the pushcart deck
    # is too small to fit 3 boxes side by side). Not tracked in the recorded state/action (which
    # is robot-only, see state_dof_indices below) - like --cube-scale variation across sessions,
    # these just diversify what the camera sees a "box" look like, for size generalization.
    #
    # Each extra box's own footprint has to be measured before its placement can be computed (so
    # it doesn't overlap the main box), so these two are placed manually rather than via
    # spawn_real_box in one call: reference -> measure at scale (scaled_footprint, non-mutating)
    # -> compute the offset from the main box's known half-width -> place_on_surface (which must
    # be called exactly once per prim, at its just-referenced identity transform - see its
    # docstring) -> mass override.
    if args.extra_boxes and args.cube_start == "table":
        main_half_dy = (main_box_aabb[4] - main_box_aabb[1]) / 2.0

        add_reference_to_stage(usd_path=assets_root_path + BOX_ASSET_CUBE2, prim_path="/World/Cube2")
        make_box_dynamic("/World/Cube2", args.cube2_mass)
        cube2_footprint = scaled_footprint(bbox_cache, "/World/Cube2", args.cube2_scale)
        cube2_half_dy = (cube2_footprint[4] - cube2_footprint[1]) / 2.0
        cube2_y = table_center_y + main_half_dy + CUBE_ROW_GAP_M + cube2_half_dy
        place_on_surface(bbox_cache, "/World/Cube2", x=table_center_x, y=cube2_y, surface_z=table_top_z, scale=args.cube2_scale)

        add_reference_to_stage(usd_path=assets_root_path + BOX_ASSET_CUBE3, prim_path="/World/Cube3")
        make_box_dynamic("/World/Cube3", args.cube3_mass)
        cube3_footprint = scaled_footprint(bbox_cache, "/World/Cube3", args.cube3_scale)
        cube3_half_dy = (cube3_footprint[4] - cube3_footprint[1]) / 2.0
        cube3_y = table_center_y - main_half_dy - CUBE_ROW_GAP_M - cube3_half_dy
        place_on_surface(bbox_cache, "/World/Cube3", x=table_center_x, y=cube3_y, surface_z=table_top_z, scale=args.cube3_scale)

        print(
            f"[geometry] cube2 (CardBoxC) half_dy={cube2_half_dy:.3f}m at y={cube2_y:.3f}  "
            f"cube3 (CardBoxA) half_dy={cube3_half_dy:.3f}m at y={cube3_y:.3f}  "
            f"(check none hang off the table edge, esp. x-axis for cube3 - Stage-panel + F on "
            f"/World/Cube2, /World/Cube3 to verify)"
        )

    camera = Camera(
        prim_path=f"{HEAD_CAMERA_MOUNT}/head_camera",
        frequency=20,
        resolution=(640, 480),
    )

    robot = SingleArticulation(ROBOT_PRIM)

    world.reset()
    robot.initialize()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()

    aperture = camera.get_horizontal_aperture()
    camera.set_focal_length(float(aperture / (2.0 * np.tan(np.radians(CAMERA_FOV_DEG) / 2.0))))

    camera_mount_prim = get_prim_at_path(HEAD_CAMERA_MOUNT)

    def get_mount_world_quat() -> np.ndarray:
        xf = UsdGeom.Xformable(camera_mount_prim).ComputeLocalToWorldTransform(0)
        q = xf.ExtractRotationQuat()
        return np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])

    def set_camera_orientation(pan_deg: float, tilt_deg: float) -> None:
        orientation = camera_pan_tilt_quat(CAMERA_ROLL_DEG, tilt_deg, pan_deg, get_mount_world_quat())
        camera.set_local_pose(translation=np.array([CAMERA_MOUNT_FORWARD_OFFSET_M, 0.0, 0.0]), orientation=orientation)

    camera_pan_deg = CAMERA_PAN_DEG
    camera_tilt_deg = CAMERA_TILT_DEG
    set_camera_orientation(camera_pan_deg, camera_tilt_deg)
    camera.set_clipping_range(near_distance=CAMERA_NEAR_CLIP_M, far_distance=CAMERA_FAR_CLIP_M)

    for _ in range(60):
        world.step(render=True)

    geometry = read_wheel_geometry(f"{ROBOT_PRIM}/OmniChassis/Graph/holonomic_controller")
    drive_controller = build_drive_controller(geometry)
    wheel_dof_indices = [robot.get_dof_index(name) for name in geometry["wheel_dof_names"]]
    remove_ros2_control_graphs(stage)

    left_arm_dof_indices = arm_dof_indices(robot, "left")
    right_arm_dof_indices = arm_dof_indices(robot, "right")
    leg_indices = leg_dof_indices(robot)
    left_gripper_dof_indices = gripper_dof_indices(robot, "left")
    right_gripper_dof_indices = gripper_dof_indices(robot, "right")
    head_dof_indices = [robot.get_dof_index("head_joint1"), robot.get_dof_index("head_joint2")]

    # For the live watchdog below - names kept in the exact same order as the indices they pair
    # with (left_gripper_dof_indices/right_gripper_dof_indices are each a single drive-joint index,
    # left_arm_dof_indices/right_arm_dof_indices are already joint1..joint7 order via
    # arm_dof_indices()).
    watchdog_names = (
        ["left_gripper_joint", "right_gripper_joint"]
        + [f"left_arm_joint{i}" for i in range(1, 8)]
        + [f"right_arm_joint{i}" for i in range(1, 8)]
    )
    watchdog_dof_indices = left_gripper_dof_indices + right_gripper_dof_indices + left_arm_dof_indices + right_arm_dof_indices
    watchdog_index_by_name = {name: i for i, name in enumerate(watchdog_names)}

    state_dof_indices = np.array(
        left_arm_dof_indices + right_arm_dof_indices + leg_indices + left_gripper_dof_indices + right_gripper_dof_indices
    )
    state_names = (
        [f"left_arm_joint{i}" for i in range(1, 8)]
        + [f"right_arm_joint{i}" for i in range(1, 8)]
        + [f"leg_joint{i}" for i in range(1, 6)]
        + ["left_gripper_joint", "right_gripper_joint"]
        + ["chassis_forward"]
    )
    # chassis_forward is not a joint - it's the 22nd (last) dim, appended separately from
    # state_dof_indices below wherever state_vec/action_vec are assembled. Its STATE value is
    # cumulative signed displacement (meters) along the chassis's own forward axis since the
    # current recording attempt started (0.0 at the first frame, reset every B-press/R-reset) -
    # NOT an absolute world position, so it stays meaningful/bounded regardless of where the
    # robot happens to be parked. Its ACTION value is the forward/back drive command that tick
    # (command[0] from compute_drive_command, roughly [-args.drive_speed, +args.drive_speed]) -
    # a velocity command, not a position target, unlike every other action dim; see CLAUDE.md.

    recorder = EpisodeRecorder(
        out_dir=args.out,
        fps=args.record_fps,
        state_names=state_names,
        camera_key="head_camera",
        image_hw=(480, 640),
        task_name=args.task,
    )
    recorder_state = RecorderState.IDLE

    print(
        f"[geometry] table_top_z={table_top_z:.3f}m  {args.place_target}_top_z={target_top_z:.3f}m  "
        f"delta={table_top_z - target_top_z:+.3f}m  cube_scale={args.cube_scale}  "
        f"(if delta is large and positive and --place-target=cart, raise --deck-riser; "
        f"see module docstring's Stage 0 check)"
    )

    # --rollout: connect now and fail fast if policy_server.py isn't reachable - a --rollout run
    # with no policy behind it is a misconfiguration, not a valid mode, so this deliberately
    # doesn't fall back to teleop silently.
    policy_client_pickup = None
    policy_client_place = None
    active_policy_client = None  # None until M (pickup) or N (place) is pressed - see on_keyboard_event
    active_task = None
    policy_action_vec = None  # last action received from the server; None until the first predict
    # chassis_forward reference frame (see state_names above) - (position, forward unit vector) at
    # the start of the current recording attempt, set on every B-press-start and R-reset. None
    # while IDLE; forward_displacement_m is 0.0 until this is set.
    episode_start_pos_xy = None
    episode_forward_dir = None
    if args.rollout:
        policy_client_pickup = PolicyClient(args.policy_host, args.policy_port)
        policy_client_pickup.connect()
        print(f"[rollout] connected to pickup policy_server.py at {args.policy_host}:{args.policy_port} (activate with M)")
        policy_client_place = PolicyClient(args.policy2_host, args.policy2_port)
        policy_client_place.connect()
        print(f"[rollout] connected to place policy_server.py at {args.policy2_host}:{args.policy2_port} (activate with N)")

    # See --starting-wrist-x-rad's help: overridable so episodes meant to extend
    # pickup_policy/place_policy's existing (pre-2026-09-08, always-joint5=0) data can match that
    # starting pose instead of silently drifting onto the newer calibrated default.
    starting_wrist_x_rad = args.starting_wrist_x_rad if args.starting_wrist_x_rad is not None else STARTING_WRIST_X_RAD
    print(
        f"[wrist] starting_wrist_x_rad={starting_wrist_x_rad:+.4f}rad "
        f"({'--starting-wrist-x-rad override' if args.starting_wrist_x_rad is not None else 'default STARTING_WRIST_X_RAD'})"
    )

    left_arm_swing_rate = arm_swing_rate("left", args.arm_speed)
    right_arm_swing_rate = arm_swing_rate("right", args.arm_speed)
    left_arm_swing_fraction = STARTING_LEFT_ARM_SWING_FRACTION
    right_arm_swing_fraction = STARTING_RIGHT_ARM_SWING_FRACTION
    torso_height_fraction = 0.0
    hand_updown_rad = STARTING_HAND_UPDOWN_RAD
    gripper_rad = 0.0
    wrist_x_rad = starting_wrist_x_rad
    wrist_y_rad = STARTING_WRIST_Y_RAD
    wrist_z_rad = STARTING_WRIST_Z_RAD

    # Live viewing only (see module docstring) - not the recorder, which samples separately at a
    # fixed rate below. No depth/lidar here, so the browser page's depth/map/point-cloud panels
    # just stay blank; harmless.
    frame_store = FrameStore()
    run_in_background(
        frame_store, host=args.host, port=args.port, static_index="collect_index.html", static_viewer_js="collect_viewer.js"
    )

    held_keys: set = set()
    reset_requested = False
    record_requested = False
    label_success_requested = False
    label_fail_requested = False
    discard_requested = False
    camera_print_requested = False
    wrist_x_print_requested = False
    wrist_y_print_requested = False
    wrist_z_print_requested = False
    activate_pickup_requested = False
    activate_place_requested = False

    def on_keyboard_event(event, *_args, **_kwargs) -> bool:
        nonlocal reset_requested, record_requested, label_success_requested, label_fail_requested, discard_requested, camera_print_requested, wrist_x_print_requested, wrist_y_print_requested, wrist_z_print_requested, activate_pickup_requested, activate_place_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                reset_requested = True
            elif event.input == carb.input.KeyboardInput.B:
                record_requested = True
            elif event.input == carb.input.KeyboardInput.Y:
                label_success_requested = True
            elif event.input == carb.input.KeyboardInput.F:
                label_fail_requested = True
            elif event.input == carb.input.KeyboardInput.BACKSPACE:
                discard_requested = True
            elif args.rollout and event.input == carb.input.KeyboardInput.M:
                # In --rollout mode grippers are policy-controlled (see the gripper block below),
                # so M/N are free to repurpose as pickup/place policy selectors instead of their
                # teleop-mode meaning (close/open grippers).
                activate_pickup_requested = True
            elif args.rollout and event.input == carb.input.KeyboardInput.N:
                activate_place_requested = True
            elif (
                event.input in DRIVE_KEY_AXES
                or event.input in TORSO_HEIGHT_KEYS
                or event.input in ARM_SWING_KEYS
                or event.input in HAND_UPDOWN_KEYS
                or (event.input in GRIPPER_KEYS and not args.rollout)
                or (event.input in WRIST_X_KEYS and not args.rollout)
                or (event.input in WRIST_Y_KEYS and not args.rollout)
                or (event.input in WRIST_Z_KEYS and not args.rollout)
                or event.input in CAMERA_ROTATE_KEYS_PAN
                or event.input in CAMERA_ROTATE_KEYS_TILT
            ):
                held_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in CAMERA_ROTATE_KEYS_PAN or event.input in CAMERA_ROTATE_KEYS_TILT:
                camera_print_requested = True
            if event.input in WRIST_X_KEYS:
                wrist_x_print_requested = True
            if event.input in WRIST_Y_KEYS:
                wrist_y_print_requested = True
            if event.input in WRIST_Z_KEYS:
                wrist_z_print_requested = True
            held_keys.discard(event.input)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("Controls: W/S drive forward/back, A/D strafe left/right, Q/E rotate, I/K torso up/down.")
    print("  Hold U: both arms swing forward (shoulder only). Hold O: swing back to open.")
    print("  Hold J: both hands raise (elbow only). Hold L: both hands lower.")
    if args.rollout:
        print(f"  M: activate PICKUP policy (task={args.task!r}). N: activate PLACE policy (task={args.task2!r}).")
        print("  Grippers/wrists are policy-controlled in --rollout mode (no manual M/N/C/V/,/./[/] control).")
    else:
        print("  Hold M: both grippers close. Hold N: both grippers open.")
        print("  Hold C/V: wrist rotate on joint5 (X). , / . : joint6 (Y). [ / ] : joint7 (Z).")
        print("  Release any wrist key to print its current angle - use it to find a good pose,")
        print("  then paste that value into STARTING_WRIST_X/Y/Z_RAD so no key press is needed.")
    print("  B: start/stop episode recording. After stop: Y=success, F=failure, Backspace=discard.")
    print("  Arrow keys: rotate the camera (Left/Right pan, Up/Down tilt) - or use the browser's")
    print("  rotate buttons. Prints pan/tilt on release - paste into CAMERA_PAN_DEG/CAMERA_TILT_DEG.")
    print("  R resets (also discards an in-progress episode). Close the window to exit.")
    print(
        f"  [watchdog] active: prints full gripper/arm joint state if any joint exceeds "
        f"{GRIPPER_WATCHDOG_VELOCITY_RAD_S} rad/s (a fling/instability event, not normal motion)."
    )
    print(
        f"  [gripper] diagnostic active: while M/N is held, prints target/actual/lead/velocity/"
        f"effort for both grippers every {GRIPPER_DIAG_PERIOD_S}s."
    )

    physics_dt = world.get_physics_dt()
    record_period = 1.0 / args.record_fps
    record_accum = 0.0
    consecutive_none_joint_frames = 0
    # Physics sim step rate (60Hz default) - used only to size the stuck-view bailout below.
    physics_hz = 1.0 / physics_dt
    watchdog_tripped = False
    gripper_diag_accum = 0.0
    compliance_tripped = {
        "joint2_left": False,
        "joint2_right": False,
        "joint4": False,
        "joint5": False,
        "joint6": False,
        "joint7": False,
    }

    while simulation_app.is_running():
        for cmd in frame_store.pop_commands():
            cmd_action = cmd.get("action")
            if cmd_action == "toggle_record":
                record_requested = True
            elif cmd_action == "label" and cmd.get("value") == "success":
                label_success_requested = True
            elif cmd_action == "label" and cmd.get("value") == "fail":
                label_fail_requested = True
            elif cmd_action == "discard":
                discard_requested = True
            elif cmd_action == "camera_rotate":
                delta = CAMERA_ROTATE_STEP_DEG * float(cmd.get("delta", 0.0))
                if cmd.get("axis") == "pan":
                    camera_pan_deg += delta
                elif cmd.get("axis") == "tilt":
                    camera_tilt_deg += delta
                set_camera_orientation(camera_pan_deg, camera_tilt_deg)
                print(
                    f"[camera] pan={camera_pan_deg:+.1f}deg tilt={camera_tilt_deg:+.1f}deg  "
                    f"(paste into CAMERA_PAN_DEG / CAMERA_TILT_DEG once you're happy)"
                )

        if camera_print_requested:
            camera_print_requested = False
            print(
                f"[camera] pan={camera_pan_deg:+.1f}deg tilt={camera_tilt_deg:+.1f}deg  "
                f"(paste into CAMERA_PAN_DEG / CAMERA_TILT_DEG once you're happy)"
            )

        if reset_requested:
            reset_requested = False
            if recorder_state is not RecorderState.IDLE:
                print("Reset requested mid-episode - discarding the buffered episode.")
                recorder.discard()
                recorder_state = RecorderState.IDLE
            world.reset()
            robot.initialize()
            # world.reset()/robot.initialize() invalidate the articulation's physics simulation
            # view; it isn't recreated until a physics step runs. One step wasn't reliably enough
            # (confirmed live: a later session still hit get_joint_positions()==None ~7 minutes
            # after a reset, well past any single-step race) - bounded-retry until it comes back,
            # same pattern as the initial 60-step warmup after the very first world.reset() above.
            for _ in range(60):
                world.step(render=True)
                if robot.get_joint_positions() is not None:
                    break
            else:
                print("[warning] physics simulation view did not come back after reset within 60 steps")
            if table2_xform is not None:
                table2_dx, table2_dy, table2_yaw_deg, table2_yaw_quat = sample_pose_jitter(
                    rng, args.table2_jitter_m, args.table2_yaw_jitter_deg
                )
                table2_xform.set_world_pose(
                    position=np.array([table2_anchor_x + table2_dx, table2_anchor_y + table2_dy, table2_anchor_z]),
                    orientation=table2_yaw_quat,
                )
                print(
                    f"[table2] episode {recorder.episode_index:04d} spawn offset dx={table2_dx:+.3f}m "
                    f"dy={table2_dy:+.3f}m yaw={table2_yaw_deg:+.1f}deg"
                )
            box_dx, box_dy, box_yaw_deg, box_yaw_quat = sample_pose_jitter(rng, args.box_jitter_m, args.box_yaw_jitter_deg)
            if args.cube_scale_min is not None:
                # Unlike the position/yaw-only branch below, scale can't be applied by just moving
                # the existing prim - place_on_surface's scale-then-measure trick only measures the
                # right footprint at a prim's just-referenced identity transform (see its
                # docstring), so this respawns /World/Cube from scratch every reset: delete, then
                # re-reference + re-place via spawn_real_box, same as the initial spawn above.
                # UNVERIFIED LIVE: deleting and re-authoring a dynamic RigidBodyAPI prim mid-session
                # (physics already running past the first world.reset()) hasn't been watched in
                # Isaac Sim - confirm PhysX actually picks up the new body (no console errors, box
                # settles/responds normally) and that the hug still converges across your chosen
                # --cube-scale-min/--cube-scale-max range before trusting this for real collection.
                cube_scale_cycle_index += 1
                box_scale = sample_cube_scale(rng, args, cube_scale_cycle_index)
                delete_prim("/World/Cube")
                spawn_real_box(
                    bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
                    x=box_center_x + box_dx, y=box_center_y + box_dy, surface_z=box_surface_z, scale=box_scale, mass=args.cube_mass,
                )
                box_xform = SingleXFormPrim("/World/Cube")
                box_xform.set_world_pose(orientation=box_yaw_quat)
                box_anchor_z = float(box_xform.get_world_pose()[0][2])
                print(
                    f"[box] episode {recorder.episode_index:04d} spawn offset dx={box_dx:+.3f}m dy={box_dy:+.3f}m "
                    f"yaw={box_yaw_deg:+.1f}deg scale={box_scale:.3f}"
                )
            else:
                box_xform.set_world_pose(
                    position=np.array([box_center_x + box_dx, box_center_y + box_dy, box_anchor_z]),
                    orientation=box_yaw_quat,
                )
                print(
                    f"[box] episode {recorder.episode_index:04d} spawn offset dx={box_dx:+.3f}m dy={box_dy:+.3f}m yaw={box_yaw_deg:+.1f}deg"
                )
            left_arm_swing_fraction = STARTING_LEFT_ARM_SWING_FRACTION
            right_arm_swing_fraction = STARTING_RIGHT_ARM_SWING_FRACTION
            torso_height_fraction = 0.0
            hand_updown_rad = STARTING_HAND_UPDOWN_RAD
            gripper_rad = 0.0
            wrist_x_rad = starting_wrist_x_rad
            wrist_y_rad = STARTING_WRIST_Y_RAD
            wrist_z_rad = STARTING_WRIST_Z_RAD
            record_accum = 0.0
            episode_start_pos_xy = None
            episode_forward_dir = None
            if args.rollout:
                policy_action_vec = None
            continue

        if activate_pickup_requested:
            activate_pickup_requested = False
            if args.rollout:
                if recorder_state is RecorderState.RECORDING and active_policy_client is policy_client_pickup:
                    # M pressed again while PICKUP is the one currently running - toggle off, same
                    # stop transition B uses, so M alone is a full start/stop pair and Y/F/Backspace
                    # still label it afterward.
                    recorder_state = RecorderState.AWAITING_LABEL
                    policy_action_vec = None
                    print(
                        f"[episode {recorder.episode_index:04d}] recording stopped "
                        f"({len(recorder.frames)} frames) - press Y (success) / F (fail) / Backspace (discard)"
                    )
                else:
                    active_policy_client = policy_client_pickup
                    active_task = args.task
                    active_policy_client.reset()
                    policy_action_vec = None
                    recorder.task_name = active_task
                    print(f"[rollout] active policy -> PICKUP (task={active_task!r})")
                    if recorder_state is RecorderState.IDLE:
                        # M alone starts the attempt too - no separate B press needed. Only
                        # auto-starts from IDLE; if a recording is already in progress under the
                        # other policy (place), M just swaps the active policy without stopping it.
                        recorder.start()
                        recorder_state = RecorderState.RECORDING
                        record_accum = 0.0
                        episode_start_pos_xy, episode_forward_dir = robot_forward_reference(robot)
                        print(f"[episode {recorder.episode_index:04d}] recording started")

        if activate_place_requested:
            activate_place_requested = False
            if args.rollout:
                if recorder_state is RecorderState.RECORDING and active_policy_client is policy_client_place:
                    # Same M/N-as-toggle behavior as PICKUP above, mirrored for symmetry.
                    recorder_state = RecorderState.AWAITING_LABEL
                    policy_action_vec = None
                    print(
                        f"[episode {recorder.episode_index:04d}] recording stopped "
                        f"({len(recorder.frames)} frames) - press Y (success) / F (fail) / Backspace (discard)"
                    )
                else:
                    active_policy_client = policy_client_place
                    active_task = args.task2
                    active_policy_client.reset()
                    policy_action_vec = None
                    recorder.task_name = active_task
                    print(f"[rollout] active policy -> PLACE (task={active_task!r})")
                    if recorder_state is RecorderState.IDLE:
                        recorder.start()
                        recorder_state = RecorderState.RECORDING
                        record_accum = 0.0
                        episode_start_pos_xy, episode_forward_dir = robot_forward_reference(robot)
                        print(f"[episode {recorder.episode_index:04d}] recording started")

        if record_requested:
            record_requested = False
            if recorder_state is RecorderState.IDLE:
                if args.rollout and active_policy_client is None:
                    print("[rollout] no policy activated yet - press M (pickup) or N (place) before B.")
                recorder.start()
                recorder_state = RecorderState.RECORDING
                record_accum = 0.0
                episode_start_pos_xy, episode_forward_dir = robot_forward_reference(robot)
                if args.rollout and active_policy_client is not None:
                    active_policy_client.reset()
                    policy_action_vec = None
                print(f"[episode {recorder.episode_index:04d}] recording started")
            elif recorder_state is RecorderState.RECORDING:
                recorder_state = RecorderState.AWAITING_LABEL
                if args.rollout:
                    # Release the chassis_forward override (see the command[0] override below)
                    # the moment recording stops, so manual W/S driving between attempts works
                    # without needing an extra M/N tap to reset it - policy_action_vec would
                    # otherwise stay frozen at its last predicted value forever, since predict()
                    # only runs while RECORDING.
                    policy_action_vec = None
                print(
                    f"[episode {recorder.episode_index:04d}] recording stopped "
                    f"({len(recorder.frames)} frames) - press Y (success) / F (fail) / Backspace (discard)"
                )

        if label_success_requested or label_fail_requested or discard_requested:
            # Y/F/Backspace now double as the "stop" action too - no separate B press needed to
            # end an attempt. Falls through into the AWAITING_LABEL handling right below with the
            # same save/discard semantics as before.
            if recorder_state is RecorderState.RECORDING:
                recorder_state = RecorderState.AWAITING_LABEL
                if args.rollout:
                    policy_action_vec = None

        if label_success_requested:
            label_success_requested = False
            if recorder_state is RecorderState.AWAITING_LABEL:
                recorder.save(success=True)
                recorder_state = RecorderState.IDLE

        if label_fail_requested:
            label_fail_requested = False
            if recorder_state is RecorderState.AWAITING_LABEL:
                recorder.save(success=False)
                recorder_state = RecorderState.IDLE

        if discard_requested:
            discard_requested = False
            if recorder_state is RecorderState.AWAITING_LABEL:
                recorder.discard()
                recorder_state = RecorderState.IDLE
                print("Episode discarded.")

        frame_store.update_status(
            {
                "state": recorder_state.name,
                "episode_index": recorder.episode_index,
                "num_frames": len(recorder.frames),
            }
        )

        command = compute_drive_command(held_keys, args.drive_speed, args.turn_speed)
        if args.rollout and policy_action_vec is not None:
            # Forward/back is policy-controlled during a rollout attempt (see chassis_forward in
            # state_names) - strafe/rotate (A/D/Q/E) stay manual, they were never part of the
            # recorded action space either way, so overriding only index 0 doesn't introduce a
            # train/inference mismatch there.
            command[0] = float(policy_action_vec[21])
        action = drive_controller.forward(command)
        robot.apply_action(ArticulationAction(joint_velocities=action.joint_velocities, joint_indices=wheel_dof_indices))

        actual_q = robot.get_joint_positions()
        if actual_q is None:
            # Physics simulation view momentarily gone - either a reset just ran (guarded against
            # above, but not always enough - confirmed live) or the viewport window is being
            # closed and simulation_app.is_running() hasn't caught up yet. Must still call
            # world.step() every iteration here - a bare `continue` was tried and confirmed live to
            # spin forever without ever stepping physics, since world.step() otherwise only runs at
            # the bottom of this loop, generating 10M+ log lines in minutes before it had to be
            # force-killed. Bail out loudly rather than spin forever if the view never comes back
            # (e.g. mid window-close, where it's expected to stay None until exit).
            world.step(render=True)
            consecutive_none_joint_frames += 1
            if consecutive_none_joint_frames > 5 * physics_hz:
                print("[error] physics simulation view has not recovered in 5s - exiting.")
                break
            continue
        consecutive_none_joint_frames = 0

        # Live instability watchdog (see GRIPPER_WATCHDOG_VELOCITY_RAD_S) - specifically added to
        # help diagnose the "hand keeps breaking when I pinch the handle and move" report: dumps
        # position/velocity/measured-effort for every gripper/arm joint the instant any of them
        # spikes past a speed no normal teleop motion should ever reach, so the exact moment and
        # which joint(s) are involved can be reported back rather than guessed from what's visible
        # on screen. Wrapped in try/except since get_measured_joint_efforts() raises (not returns
        # None) if the physics handle isn't valid - shouldn't happen this soon after the actual_q
        # None-check above, but a diagnostic feature crashing the whole session would be worse than
        # it silently skipping one frame.
        try:
            watchdog_qd = robot.get_joint_velocities(joint_indices=watchdog_dof_indices)
            watchdog_effort = robot.get_measured_joint_efforts(joint_indices=watchdog_dof_indices)
        except Exception:
            watchdog_qd = None
            watchdog_effort = None
        if watchdog_qd is not None:
            spike_mask = np.abs(watchdog_qd) > GRIPPER_WATCHDOG_VELOCITY_RAD_S
            if np.any(spike_mask) and not watchdog_tripped:
                watchdog_tripped = True
                watchdog_pos = actual_q[watchdog_dof_indices]
                print(f"[watchdog] INSTABILITY DETECTED - joint velocity exceeded {GRIPPER_WATCHDOG_VELOCITY_RAD_S} rad/s:")
                for i, name in enumerate(watchdog_names):
                    if spike_mask[i]:
                        effort_str = f"{watchdog_effort[i]:+.4f}Nm" if watchdog_effort is not None else "n/a"
                        print(f"    {name}: pos={watchdog_pos[i]:+.4f}rad vel={watchdog_qd[i]:+.4f}rad/s effort={effort_str}")
            elif not np.any(spike_mask) and watchdog_tripped:
                watchdog_tripped = False
                print("[watchdog] cleared - joint velocities back to normal.")

        # Effort-based compliance (see JOINT_EFFORT_LIMIT_NM) - reuses the same effort reading the
        # watchdog above just took, no extra API call. Returns False (never blocks) if the reading
        # failed this frame, matching the watchdog's own fail-open behavior right above.
        def joint_overloaded(name: str) -> bool:
            if watchdog_effort is None:
                return False
            return bool(abs(watchdog_effort[watchdog_index_by_name[name]]) > JOINT_EFFORT_LIMIT_NM)

        def report_compliance(key: str, is_overloaded: bool, label: str) -> None:
            # Print only on state transitions, not every frame - mirrors watchdog_tripped's
            # pattern. Without this, a frozen key looks like an unexplained input bug rather than
            # the deliberate compliance behavior it is.
            if is_overloaded and not compliance_tripped[key]:
                compliance_tripped[key] = True
                print(f"[compliance] {label} overloaded (>{JOINT_EFFORT_LIMIT_NM}Nm) - further motion into load blocked.")
            elif not is_overloaded and compliance_tripped[key]:
                compliance_tripped[key] = False
                print(f"[compliance] {label} cleared - back to normal control.")

        if args.rollout:
            # policy_action_vec is the last full 21-dim vector received from policy_server.py (see
            # the record_accum-gated query below) - None until the first prediction of an attempt
            # arrives, during which the arms hold the exact same STARTING_*-fraction pose teleop
            # mode's held_keys loop initializes left_arm_swing_fraction/right_arm_swing_fraction/
            # hand_updown_rad to (NOT the fully-open pose - that's a materially different, more
            # retracted pose, and defaulting to it here made the robot look frozen at launch since
            # it's close to the raw spawn pose, with none of the visible ~5s settle-into-position
            # motion teleop mode shows).
            if policy_action_vec is not None:
                left_arm_q = policy_action_vec[0:7].copy()
                right_arm_q = policy_action_vec[7:14].copy()
            else:
                left_arm_q = (1.0 - STARTING_LEFT_ARM_SWING_FRACTION) * np.array(
                    ARM_OPEN_POSE["left"]
                ) + STARTING_LEFT_ARM_SWING_FRACTION * np.array(ARM_FORWARD_POSE["left"])
                right_arm_q = (1.0 - STARTING_RIGHT_ARM_SWING_FRACTION) * np.array(
                    ARM_OPEN_POSE["right"]
                ) + STARTING_RIGHT_ARM_SWING_FRACTION * np.array(ARM_FORWARD_POSE["right"])
                left_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += STARTING_HAND_UPDOWN_RAD
                right_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += STARTING_HAND_UPDOWN_RAD
                left_arm_q[WRIST_X_JOINT_INDEX] += starting_wrist_x_rad
                right_arm_q[WRIST_X_JOINT_INDEX] += starting_wrist_x_rad
                left_arm_q[WRIST_Y_JOINT_INDEX] += STARTING_WRIST_Y_RAD
                right_arm_q[WRIST_Y_JOINT_INDEX] += STARTING_WRIST_Y_RAD
                left_arm_q[WRIST_Z_JOINT_INDEX] += STARTING_WRIST_Z_RAD
                right_arm_q[WRIST_Z_JOINT_INDEX] += STARTING_WRIST_Z_RAD
        else:
            # Compliance: block only the FORWARD (U, direction>0 - toward ARM_FORWARD_POSE, i.e.
            # further into whatever it's pressed against) increment on a side whose own joint2 is
            # under high sustained load - O (retreat) always still works regardless, so the escape
            # path out of an overloaded contact is never blocked.
            left_joint2_overloaded = joint_overloaded("left_arm_joint2")
            right_joint2_overloaded = joint_overloaded("right_arm_joint2")
            report_compliance("joint2_left", left_joint2_overloaded, "left arm swing (joint2)")
            report_compliance("joint2_right", right_joint2_overloaded, "right arm swing (joint2)")
            for key in held_keys:
                if key in ARM_SWING_KEYS:
                    direction = ARM_SWING_KEYS[key]
                    if not (direction > 0 and left_joint2_overloaded):
                        left_arm_swing_fraction += direction * left_arm_swing_rate * physics_dt
                    if not (direction > 0 and right_joint2_overloaded):
                        right_arm_swing_fraction += direction * right_arm_swing_rate * physics_dt
            left_arm_swing_fraction = float(np.clip(left_arm_swing_fraction, 0.0, 1.0))
            right_arm_swing_fraction = float(np.clip(right_arm_swing_fraction, 0.0, 1.0))
            left_arm_q = (1.0 - left_arm_swing_fraction) * np.array(ARM_OPEN_POSE["left"]) + left_arm_swing_fraction * np.array(
                ARM_FORWARD_POSE["left"]
            )
            right_arm_q = (1.0 - right_arm_swing_fraction) * np.array(
                ARM_OPEN_POSE["right"]
            ) + right_arm_swing_fraction * np.array(ARM_FORWARD_POSE["right"])

            # Compliance: unlike swing above, there's no clear "safe" direction for the elbow under
            # load (which way relieves force depends on what it's pushing against), so this freezes
            # both directions entirely once either arm's joint4 is overloaded, rather than guessing.
            joint4_overloaded = joint_overloaded("left_arm_joint4") or joint_overloaded("right_arm_joint4")
            report_compliance("joint4", joint4_overloaded, "elbow (joint4)")
            if not joint4_overloaded:
                for key in held_keys:
                    if key in HAND_UPDOWN_KEYS:
                        hand_updown_rad += HAND_UPDOWN_KEYS[key] * args.arm_speed * physics_dt
            hand_updown_rad = float(np.clip(hand_updown_rad, -ARM_HAND_DOWN_MAX_RAD, ARM_HAND_UP_MAX_RAD))
            left_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += hand_updown_rad
            right_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += hand_updown_rad

            # Compliance: same "freeze both directions, no safe-direction guess" reasoning as
            # joint4 above - this is also the specific joint (wrist-rotate/joint5) that's actually
            # shown the largest live effort spikes so far (-66.2Nm/-75.1Nm across two separate
            # pinch-and-push tests), making this the highest-value place this mechanism applies.
            joint5_overloaded = joint_overloaded("left_arm_joint5") or joint_overloaded("right_arm_joint5")
            report_compliance("joint5", joint5_overloaded, "wrist rotate X (joint5)")
            if not joint5_overloaded:
                for key in held_keys:
                    if key in WRIST_X_KEYS:
                        wrist_x_rad += WRIST_X_KEYS[key] * args.arm_speed * physics_dt
            wrist_x_rad = float(np.clip(wrist_x_rad, WRIST_X_MIN_RAD, WRIST_X_MAX_RAD))
            if wrist_x_print_requested:
                # Same "print the value on key release, paste it back into the constant" pattern
                # as the camera pan/tilt calibration above - neither STARTING_WRIST_X_RAD nor
                # STARTING_WRIST_Y_RAD can be derived analytically (see their own comment), so
                # these keys plus this readout are the actual way to find them: hold the axis's
                # key pair until the gripper reads the pose you want, then copy the printed
                # radians into the matching constant.
                wrist_x_print_requested = False
                print(f"[wrist] joint5={wrist_x_rad:+.4f}rad ({np.degrees(wrist_x_rad):+.1f}deg)")
            left_arm_q[WRIST_X_JOINT_INDEX] += wrist_x_rad
            right_arm_q[WRIST_X_JOINT_INDEX] += wrist_x_rad

            joint6_overloaded = joint_overloaded("left_arm_joint6") or joint_overloaded("right_arm_joint6")
            report_compliance("joint6", joint6_overloaded, "wrist rotate Y (joint6)")
            if not joint6_overloaded:
                for key in held_keys:
                    if key in WRIST_Y_KEYS:
                        wrist_y_rad += WRIST_Y_KEYS[key] * args.arm_speed * physics_dt
            wrist_y_rad = float(np.clip(wrist_y_rad, WRIST_Y_MIN_RAD, WRIST_Y_MAX_RAD))
            if wrist_y_print_requested:
                wrist_y_print_requested = False
                print(f"[wrist] joint6={wrist_y_rad:+.4f}rad ({np.degrees(wrist_y_rad):+.1f}deg)")
            left_arm_q[WRIST_Y_JOINT_INDEX] += wrist_y_rad
            right_arm_q[WRIST_Y_JOINT_INDEX] += wrist_y_rad

            joint7_overloaded = joint_overloaded("left_arm_joint7") or joint_overloaded("right_arm_joint7")
            report_compliance("joint7", joint7_overloaded, "wrist rotate Z (joint7)")
            if not joint7_overloaded:
                for key in held_keys:
                    if key in WRIST_Z_KEYS:
                        wrist_z_rad += WRIST_Z_KEYS[key] * args.arm_speed * physics_dt
            wrist_z_rad = float(np.clip(wrist_z_rad, WRIST_Z_MIN_RAD, WRIST_Z_MAX_RAD))
            if wrist_z_print_requested:
                wrist_z_print_requested = False
                print(f"[wrist] joint7={wrist_z_rad:+.4f}rad ({np.degrees(wrist_z_rad):+.1f}deg)")
            left_arm_q[WRIST_Z_JOINT_INDEX] += wrist_z_rad
            right_arm_q[WRIST_Z_JOINT_INDEX] += wrist_z_rad

        # Safety-critical: this clamp (and the matching ones for torso/grippers below) is what
        # guards against the joint-velocity-spike/fling failure mode documented in CLAUDE.md - it
        # applies identically whether the pre-clamp target above came from a held key or a policy
        # prediction, on purpose. Never let a --rollout target bypass this.
        left_arm_q = clamp_to_actual(left_arm_q, actual_q[left_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        right_arm_q = clamp_to_actual(right_arm_q, actual_q[right_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        robot.apply_action(ArticulationAction(joint_positions=left_arm_q, joint_indices=left_arm_dof_indices))
        robot.apply_action(ArticulationAction(joint_positions=right_arm_q, joint_indices=right_arm_dof_indices))

        if args.rollout:
            torso_q = policy_action_vec[14:19].copy() if policy_action_vec is not None else np.array(TORSO_UP_POSE)
        else:
            for key in held_keys:
                if key in TORSO_HEIGHT_KEYS:
                    torso_height_fraction += TORSO_HEIGHT_KEYS[key] * args.torso_speed * physics_dt
            torso_height_fraction = float(np.clip(torso_height_fraction, 0.0, 1.0))
            torso_q = (1.0 - torso_height_fraction) * np.array(TORSO_UP_POSE) + torso_height_fraction * np.array(TORSO_DOWN_POSE)
        torso_q = clamp_to_actual(torso_q, actual_q[leg_indices])
        robot.apply_action(ArticulationAction(joint_positions=torso_q, joint_indices=leg_indices))

        if args.rollout:
            if policy_action_vec is not None:
                left_gripper_target = policy_action_vec[19:20].copy()
                right_gripper_target = policy_action_vec[20:21].copy()
            else:
                left_gripper_target = np.array([0.0])
                right_gripper_target = np.array([0.0])
        else:
            for key in held_keys:
                if key in GRIPPER_KEYS:
                    gripper_rad += GRIPPER_KEYS[key] * GRIPPER_SPEED_RAD_S * physics_dt
            gripper_rad = float(np.clip(gripper_rad, 0.0, GRIPPER_CLOSE_MAX_RAD))
            # Re-sync to the actual joint position every step, not just clip to [0, max] - without
            # this, gripper_rad (advancing at GRIPPER_SPEED_RAD_S=2.5 rad/s while M/N is held) races
            # far ahead of the real joint, which can only track it at roughly
            # GRIPPER_MAX_LEAD_RAD/physics_dt (~0.5 rad/s, deliberately slow - see that constant's
            # comment). Releasing the key then did nothing to stop the motion: the joint just kept
            # creeping toward that stale, far-ahead target at its lead-clamped rate, including
            # grinding into whatever the fingers had already touched. Clamping gripper_rad itself
            # (not just the final applied target further below) to stay near the actual position
            # makes releasing M/N stop the motion immediately, and makes contact resistance (actual
            # position stalling against something) stop further closing instead of continuing to
            # close against a backlog that was never visible to the user.
            gripper_actual = float(np.mean([actual_q[left_gripper_dof_indices[0]], actual_q[right_gripper_dof_indices[0]]]))
            gripper_rad = float(np.clip(gripper_rad, gripper_actual - GRIPPER_MAX_LEAD_RAD, gripper_actual + GRIPPER_MAX_LEAD_RAD))
            left_gripper_target = np.array([gripper_rad])
            right_gripper_target = np.array([gripper_rad])
        left_gripper_q = clamp_to_actual(left_gripper_target, actual_q[left_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
        right_gripper_q = clamp_to_actual(right_gripper_target, actual_q[right_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
        robot.apply_action(ArticulationAction(joint_positions=left_gripper_q, joint_indices=left_gripper_dof_indices))
        robot.apply_action(ArticulationAction(joint_positions=right_gripper_q, joint_indices=right_gripper_dof_indices))

        # Gripper-specific diagnostic trace (see GRIPPER_DIAG_PERIOD_S) - only while M/N is
        # actually held, at a fixed print rate rather than every physics step. watchdog_qd/
        # watchdog_effort were already read this frame (see the watchdog block above) with
        # left/right gripper as indices 0/1 of that same array - reused here rather than a second
        # API call for the same data.
        if not args.rollout and any(key in held_keys for key in GRIPPER_KEYS):
            gripper_diag_accum += physics_dt
            if gripper_diag_accum >= GRIPPER_DIAG_PERIOD_S:
                gripper_diag_accum = 0.0
                left_actual = float(actual_q[left_gripper_dof_indices[0]])
                right_actual = float(actual_q[right_gripper_dof_indices[0]])
                left_vel = float(watchdog_qd[0]) if watchdog_qd is not None else float("nan")
                right_vel = float(watchdog_qd[1]) if watchdog_qd is not None else float("nan")
                left_eff = float(watchdog_effort[0]) if watchdog_effort is not None else float("nan")
                right_eff = float(watchdog_effort[1]) if watchdog_effort is not None else float("nan")
                print(
                    f"[gripper] target={gripper_rad:+.4f}rad  "
                    f"L: actual={left_actual:+.4f} lead={gripper_rad - left_actual:+.4f} vel={left_vel:+.4f}rad/s effort={left_eff:+.4f}Nm  "
                    f"R: actual={right_actual:+.4f} lead={gripper_rad - right_actual:+.4f} vel={right_vel:+.4f}rad/s effort={right_eff:+.4f}Nm"
                )
        else:
            gripper_diag_accum = 0.0

        camera_rotation_changed = False
        for key in held_keys:
            if key in CAMERA_ROTATE_KEYS_PAN:
                camera_pan_deg += CAMERA_ROTATE_KEYS_PAN[key] * CAMERA_ROTATE_KEY_SPEED_DEG_S * physics_dt
                camera_rotation_changed = True
            if key in CAMERA_ROTATE_KEYS_TILT:
                camera_tilt_deg += CAMERA_ROTATE_KEYS_TILT[key] * CAMERA_ROTATE_KEY_SPEED_DEG_S * physics_dt
                camera_rotation_changed = True
        if camera_rotation_changed:
            # Applied every step while held (for live visual feedback) but only printed on key
            # release (see camera_print_requested) - printing at 60Hz while a key is held would
            # spam the console.
            set_camera_orientation(camera_pan_deg, camera_tilt_deg)

        hold_head_joints(robot, head_dof_indices)

        world.step(render=True)

        rgba = camera.get_rgba()
        if rgba is not None:
            frame_store.update_rgb(rgba)
        depth = camera.get_depth()
        if depth is not None:
            frame_store.update_depth(depth)

        record_accum += physics_dt
        if recorder_state is RecorderState.RECORDING and record_accum >= record_period:
            record_accum -= record_period
            chassis_forward_state = forward_displacement(robot, episode_start_pos_xy, episode_forward_dir)
            state_vec = np.concatenate(
                [robot.get_joint_positions()[state_dof_indices], [chassis_forward_state]]
            ).astype(np.float32)
            if args.rollout and rgba is not None and active_policy_client is not None:
                # Query at the same record_fps cadence the policy was trained at. This is the
                # observation-to-action edge of the loop: the frame/state captured just above
                # becomes the target used by the joint-control block above on the *next* iteration
                # (one physics step of latency, ~1/60s at default settings - negligible, and no
                # different in kind from any real inference pipeline's latency).
                policy_action_vec = active_policy_client.predict(rgba[:, :, :3], state_vec, active_task)
            if rgba is not None and depth is not None:
                action_vec = np.concatenate(
                    [left_arm_q, right_arm_q, torso_q, left_gripper_q, right_gripper_q, [command[0]]]
                ).astype(np.float32)
                recorder.append(rgba, depth, state_vec, action_vec)

    if policy_client_pickup is not None:
        policy_client_pickup.close()
    if policy_client_place is not None:
        policy_client_place.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
