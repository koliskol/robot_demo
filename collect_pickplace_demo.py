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
parser.add_argument("--out", type=str, default="./raw_episodes", help="Output directory for recorded episodes.")
parser.add_argument("--record-fps", type=float, default=15.0, help="Fixed sample rate for recorded episodes.")
parser.add_argument("--task", type=str, default=None, help="Task name stored in each episode's manifest (default: derived from --cube-start).")
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
    default=0.15,
    help="Chassis rotation command magnitude. Lower than stream_demo.py's 0.3 default, same "
    "tip-over reasoning as --drive-speed.",
)
parser.add_argument("--arm-speed", type=float, default=0.4, help="Max arm joint speed in radians/second.")
parser.add_argument("--torso-speed", type=float, default=0.4, help="Torso up/down speed, as a fraction/second of its full travel.")
args = parser.parse_args()
if args.cube_start not in ("table", args.place_target):
    parser.error(f"--cube-start {args.cube_start!r} requires --place-target {args.cube_start!r} (got --place-target {args.place_target!r})")
if args.task is None:
    args.task = f"pick_box_table_to_{args.place_target}" if args.cube_start == "table" else f"pick_box_{args.place_target}_to_table"

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
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

from streaming_server import FrameStore, run_in_background

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
# height as the main one, placed adjacent to it (same near-edge-flush pattern as the cart, see
# ROBOT_APPROACH_GAP_M's comment above). TABLE2_GAP_M mirrors CART_TABLE_GAP_M's role. Table2 uses
# the full-size table asset (0.8 x 2.8m, same as table1), so unlike the small pushcart deck its own
# centroid is far outside a parked robot's reach - TABLE2_EDGE_INSET_M places the actual pick/place
# point a short distance onto table2's surface from its near edge instead, analogous to how the
# small cart deck is reachable almost in its entirety.
TABLE2_GAP_M = 0.15
TABLE2_EDGE_INSET_M = 0.3

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

GRIPPER_CLOSE_MAX_RAD = np.radians(97.57470703125)
GRIPPER_SPEED_RAD_S = 2.5
GRIPPER_KEYS = {
    carb.input.KeyboardInput.N: -1.0,  # toward open
    carb.input.KeyboardInput.M: 1.0,  # toward closed
}
GRIPPER_MAX_LEAD_RAD = 0.008

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
PUSHCART_DECK_HALF_EXTENT = (0.45, 0.225)  # width (x) widened from 0.3 - see TUTORIAL.md's "0.6m width" note
PUSHCART_DECK_THICKNESS = 0.03
PUSHCART_WHEEL_RADIUS = 0.05
PUSHCART_HANDLE_POST_HEIGHT = 0.75
PUSHCART_CHASSIS_MASS = 4.4
PUSHCART_FORK_MASS = 0.05
PUSHCART_WHEEL_MASS = 0.1
CASTER_ROLLING_FRICTION_NM = 0.05


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
    frictionless or welded, etc.). Identical to that version except deck_riser_height, which
    raises the deck (and its collision box) above the stock ~0.15m height without touching the
    already-stability-tuned caster fork/wheel joint geometry.
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

        wheel = UsdGeom.Cylinder.Define(stage, f"{prim_path}/Wheel{i}")
        wheel.CreateRadiusAttr(PUSHCART_WHEEL_RADIUS)
        wheel.CreateHeightAttr(0.03)
        wheel.CreateAxisAttr("Y")
        wheel.AddTranslateOp().Set(wheel_center)
        wheel.CreateDisplayColorAttr([(0.05, 0.05, 0.05)])
        UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
        UsdPhysics.MassAPI.Apply(wheel.GetPrim()).CreateMassAttr(PUSHCART_WHEEL_MASS)

        spin = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/CasterSpin{i}")
        spin.CreateBody0Rel().SetTargets([Sdf.Path(f"{prim_path}/CasterFork{i}")])
        spin.CreateBody1Rel().SetTargets([Sdf.Path(f"{prim_path}/Wheel{i}")])
        spin.CreateAxisAttr("Y")
        spin.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        spin.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        PhysxSchema.PhysxJointAPI.Apply(spin.GetPrim()).CreateJointFrictionAttr(CASTER_ROLLING_FRICTION_NM)

    post_x = -dx + 0.02
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

    # Pick/place partner - either the pushcart or a second table, chosen via --place-target (see
    # that flag's help and TABLE2_GAP_M's comment above). Only one is ever built - they're
    # alternative task variants, not simultaneous targets (a single parked robot pose can't reach
    # both the cart and a full-size table2 at once).
    if args.place_target == "cart":
        # Cart placed beside the table, front edge flush with the table's own near (-x) edge, so
        # both sit at the same approach depth and only differ in y - see ROBOT_APPROACH_GAP_M's
        # comment above for why this exact layout is a first guess, not a verified one.
        dx, dy = PUSHCART_DECK_HALF_EXTENT
        target_x = table_aabb[0] + dx
        target_y = table_aabb[4] + CART_TABLE_GAP_M + dy
        build_pushcart(stage, "/World/PushCart", x=target_x, y=target_y, deck_riser_height=args.deck_riser)
        target_top_z = pushcart_deck_top_z(args.deck_riser)
    else:
        table2_half_dx = (table_aabb[3] - table_aabb[0]) / 2.0
        table2_half_dy = (table_aabb[4] - table_aabb[1]) / 2.0
        add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table2")
        table2_aabb = place_on_ground(
            bbox_cache, "/World/Table2",
            x=table_aabb[0] + table2_half_dx, y=table_aabb[4] + TABLE2_GAP_M + table2_half_dy,
            z_scale=args.table_height_scale,
        )
        target_x = table_center_x
        target_y = table_aabb[4] + TABLE2_GAP_M + TABLE2_EDGE_INSET_M
        target_top_z = table2_aabb[5]

    add_reference_to_stage(usd_path=assets_root_path + ROBOT_ASSET, prim_path=ROBOT_PRIM)
    stiffen_head_joints()
    robot_spawn_x = table_aabb[0] - ROBOT_APPROACH_GAP_M
    robot_spawn_y = (table_center_y + target_y) / 2.0  # centered between table and the place-target
    place_on_ground(bbox_cache, ROBOT_PRIM, x=robot_spawn_x, y=robot_spawn_y)

    if args.cube_start == "table":
        main_box_aabb = spawn_real_box(
            bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
            x=table_center_x, y=table_center_y, surface_z=table_top_z, scale=args.cube_scale, mass=args.cube_mass,
        )
    else:
        main_box_aabb = spawn_real_box(
            bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
            x=target_x, y=target_y, surface_z=target_top_z, scale=args.cube_scale, mass=args.cube_mass,
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

    state_dof_indices = np.array(
        left_arm_dof_indices + right_arm_dof_indices + leg_indices + left_gripper_dof_indices + right_gripper_dof_indices
    )
    state_names = (
        [f"left_arm_joint{i}" for i in range(1, 8)]
        + [f"right_arm_joint{i}" for i in range(1, 8)]
        + [f"leg_joint{i}" for i in range(1, 6)]
        + ["left_gripper_joint", "right_gripper_joint"]
    )

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

    left_arm_swing_rate = arm_swing_rate("left", args.arm_speed)
    right_arm_swing_rate = arm_swing_rate("right", args.arm_speed)
    left_arm_swing_fraction = STARTING_LEFT_ARM_SWING_FRACTION
    right_arm_swing_fraction = STARTING_RIGHT_ARM_SWING_FRACTION
    torso_height_fraction = 0.0
    hand_updown_rad = STARTING_HAND_UPDOWN_RAD
    gripper_rad = 0.0

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

    def on_keyboard_event(event, *_args, **_kwargs) -> bool:
        nonlocal reset_requested, record_requested, label_success_requested, label_fail_requested, discard_requested, camera_print_requested
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
            elif (
                event.input in DRIVE_KEY_AXES
                or event.input in TORSO_HEIGHT_KEYS
                or event.input in ARM_SWING_KEYS
                or event.input in HAND_UPDOWN_KEYS
                or event.input in GRIPPER_KEYS
                or event.input in CAMERA_ROTATE_KEYS_PAN
                or event.input in CAMERA_ROTATE_KEYS_TILT
            ):
                held_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in CAMERA_ROTATE_KEYS_PAN or event.input in CAMERA_ROTATE_KEYS_TILT:
                camera_print_requested = True
            held_keys.discard(event.input)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("Controls: W/S drive forward/back, A/D strafe left/right, Q/E rotate, I/K torso up/down.")
    print("  Hold U: both arms swing forward (shoulder only). Hold O: swing back to open.")
    print("  Hold J: both hands raise (elbow only). Hold L: both hands lower.")
    print("  Hold M: both grippers close. Hold N: both grippers open.")
    print("  B: start/stop episode recording. After stop: Y=success, F=failure, Backspace=discard.")
    print("  Arrow keys: rotate the camera (Left/Right pan, Up/Down tilt) - or use the browser's")
    print("  rotate buttons. Prints pan/tilt on release - paste into CAMERA_PAN_DEG/CAMERA_TILT_DEG.")
    print("  R resets (also discards an in-progress episode). Close the window to exit.")

    physics_dt = world.get_physics_dt()
    record_period = 1.0 / args.record_fps
    record_accum = 0.0

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
            left_arm_swing_fraction = STARTING_LEFT_ARM_SWING_FRACTION
            right_arm_swing_fraction = STARTING_RIGHT_ARM_SWING_FRACTION
            torso_height_fraction = 0.0
            hand_updown_rad = STARTING_HAND_UPDOWN_RAD
            gripper_rad = 0.0
            record_accum = 0.0
            continue

        if record_requested:
            record_requested = False
            if recorder_state is RecorderState.IDLE:
                recorder.start()
                recorder_state = RecorderState.RECORDING
                record_accum = 0.0
                print(f"[episode {recorder.episode_index:04d}] recording started")
            elif recorder_state is RecorderState.RECORDING:
                recorder_state = RecorderState.AWAITING_LABEL
                print(
                    f"[episode {recorder.episode_index:04d}] recording stopped "
                    f"({len(recorder.frames)} frames) - press Y (success) / F (fail) / Backspace (discard)"
                )

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
        action = drive_controller.forward(command)
        robot.apply_action(ArticulationAction(joint_velocities=action.joint_velocities, joint_indices=wheel_dof_indices))

        actual_q = robot.get_joint_positions()

        for key in held_keys:
            if key in ARM_SWING_KEYS:
                left_arm_swing_fraction += ARM_SWING_KEYS[key] * left_arm_swing_rate * physics_dt
                right_arm_swing_fraction += ARM_SWING_KEYS[key] * right_arm_swing_rate * physics_dt
        left_arm_swing_fraction = float(np.clip(left_arm_swing_fraction, 0.0, 1.0))
        right_arm_swing_fraction = float(np.clip(right_arm_swing_fraction, 0.0, 1.0))
        left_arm_q = (1.0 - left_arm_swing_fraction) * np.array(ARM_OPEN_POSE["left"]) + left_arm_swing_fraction * np.array(
            ARM_FORWARD_POSE["left"]
        )
        right_arm_q = (1.0 - right_arm_swing_fraction) * np.array(
            ARM_OPEN_POSE["right"]
        ) + right_arm_swing_fraction * np.array(ARM_FORWARD_POSE["right"])

        for key in held_keys:
            if key in HAND_UPDOWN_KEYS:
                hand_updown_rad += HAND_UPDOWN_KEYS[key] * args.arm_speed * physics_dt
        hand_updown_rad = float(np.clip(hand_updown_rad, -ARM_HAND_DOWN_MAX_RAD, ARM_HAND_UP_MAX_RAD))
        left_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += hand_updown_rad
        right_arm_q[ARM_HAND_UPDOWN_JOINT_INDEX] += hand_updown_rad

        left_arm_q = clamp_to_actual(left_arm_q, actual_q[left_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        right_arm_q = clamp_to_actual(right_arm_q, actual_q[right_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        robot.apply_action(ArticulationAction(joint_positions=left_arm_q, joint_indices=left_arm_dof_indices))
        robot.apply_action(ArticulationAction(joint_positions=right_arm_q, joint_indices=right_arm_dof_indices))

        for key in held_keys:
            if key in TORSO_HEIGHT_KEYS:
                torso_height_fraction += TORSO_HEIGHT_KEYS[key] * args.torso_speed * physics_dt
        torso_height_fraction = float(np.clip(torso_height_fraction, 0.0, 1.0))
        torso_q = (1.0 - torso_height_fraction) * np.array(TORSO_UP_POSE) + torso_height_fraction * np.array(TORSO_DOWN_POSE)
        torso_q = clamp_to_actual(torso_q, actual_q[leg_indices])
        robot.apply_action(ArticulationAction(joint_positions=torso_q, joint_indices=leg_indices))

        for key in held_keys:
            if key in GRIPPER_KEYS:
                gripper_rad += GRIPPER_KEYS[key] * GRIPPER_SPEED_RAD_S * physics_dt
        gripper_rad = float(np.clip(gripper_rad, 0.0, GRIPPER_CLOSE_MAX_RAD))
        left_gripper_q = clamp_to_actual(np.array([gripper_rad]), actual_q[left_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
        right_gripper_q = clamp_to_actual(np.array([gripper_rad]), actual_q[right_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
        robot.apply_action(ArticulationAction(joint_positions=left_gripper_q, joint_indices=left_gripper_dof_indices))
        robot.apply_action(ArticulationAction(joint_positions=right_gripper_q, joint_indices=right_gripper_dof_indices))

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
            if rgba is not None and depth is not None:
                state_vec = robot.get_joint_positions()[state_dof_indices].astype(np.float32)
                action_vec = np.concatenate([left_arm_q, right_arm_q, torso_q, left_gripper_q, right_gripper_q]).astype(
                    np.float32
                )
                recorder.append(rgba, depth, state_vec, action_vec)

    simulation_app.close()


if __name__ == "__main__":
    main()
