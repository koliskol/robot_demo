"""Drive a Galbot G1 mobile robot through a table<->pushcart pick-and-place task in Isaac Sim,
recording keyboard-teleoperated demonstrations to disk for later conversion into a LeRobot
dataset (see convert_to_lerobot.py, run in a separate lerobot-installed environment - this
script deliberately never imports lerobot itself, to keep it out of the isaac_sim conda env's
dependency footprint).

Usage:

    conda run -n isaac_sim python collect_pickplace_demo.py
    conda run -n isaac_sim python collect_pickplace_demo.py --out ./raw_episodes --deck-riser 0.5

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

    SPACE       toggle: start recording an episode / stop and await a label
    Y           (after SPACE-stop) label the just-recorded episode a SUCCESS and save it
    F           (after SPACE-stop) label the just-recorded episode a FAILURE and save it
    BACKSPACE   (after SPACE-stop) discard the just-recorded episode without saving

    Close the viewport window to exit.

IMPORTANT - read before collecting any real data: this task requires holding and carrying a loose
object, which neither this project nor ../Robot_project/capture_cube_rgbd.py has ever
demonstrated. The chosen approach is a bimanual "hug" - both arms swinging forward (U) to
compress the box between the forearms, rather than a single gripper's fingertip pinch - so the
boxes (real YCB box assets, see BOX_ASSET_MAIN/BOX_ASSET_EXTRA below - not procedural cubes) are
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

Camera/lidar mounting and the arm/hand/torso jog constants mirror stream_demo.py and
../Robot_project/capture_cube_rgbd.py exactly (same Galbot G1 asset, same joint targets/clamps/
rates) - see stream_demo.py's module docstring for the full derivation. This script drops the
WebRTC streaming and lidar/depth entirely (not needed for offline data collection) and adds a
pushcart + a graspable cube, ported/adapted from capture_cube_rgbd.py's build_pushcart and cube
spawn (see those functions below for what changed and why).
"""

import argparse
import json
from enum import Enum, auto
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=str, default="./raw_episodes", help="Output directory for recorded episodes.")
parser.add_argument("--record-fps", type=float, default=15.0, help="Fixed sample rate for recorded episodes.")
parser.add_argument("--task", type=str, default=None, help="Task name stored in each episode's manifest (default: derived from --cube-start).")
parser.add_argument(
    "--cube-start",
    type=str,
    choices=["table", "cart"],
    default="table",
    help="Where the cube spawns on reset - run one session per direction to collect both "
    "table->cart and cart->table demonstrations.",
)
parser.add_argument(
    "--cube-scale",
    type=float,
    default=1.0,
    help="Uniform scale multiplier for the main box (a real cracker-box asset, see BOX_ASSET_MAIN "
    "- native footprint is roughly 0.16 x 0.21 x 0.07m at scale 1.0).",
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
    default=1.5,
    help="Scale multiplier for a second, bigger box (a real sugar-box asset, see BOX_ASSET_EXTRA "
    "- native footprint is roughly 0.09 x 0.18 x 0.05m at scale 1.0). Table-side only.",
)
parser.add_argument("--cube2-mass", type=float, default=0.22, help="Mass of the second box in kg.")
parser.add_argument(
    "--cube3-scale",
    type=float,
    default=1.3,
    help="Scale multiplier for a third, even bigger box (reuses BOX_ASSET_MAIN, the cracker-box "
    "asset, at a larger scale than the main box). Table-side only.",
)
parser.add_argument("--cube3-mass", type=float, default=0.30, help="Mass of the third box in kg.")
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
    "--deck-riser",
    type=float,
    default=0.0,
    help="Extra meters added to the pushcart deck's stock height (~0.15m) - tune this after the Stage 0 "
    "manual reach check described in the module docstring; raise it if the gripper can't get low enough "
    "over the deck to release the cube.",
)
parser.add_argument("--drive-speed", type=float, default=1.0, help="Chassis drive/strafe command magnitude.")
parser.add_argument("--turn-speed", type=float, default=0.3, help="Chassis rotation command magnitude.")
parser.add_argument("--arm-speed", type=float, default=0.4, help="Max arm joint speed in radians/second.")
parser.add_argument("--torso-speed", type=float, default=0.4, help="Torso up/down speed, as a fraction/second of its full travel.")
args = parser.parse_args()
if args.task is None:
    args.task = "pick_box_table_to_cart" if args.cube_start == "table" else "pick_box_cart_to_table"

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

TABLE_ASSET = "/Isaac/Environments/Office/Props/SM_TableB.usd"
ROBOT_ASSET = "/Isaac/Robots/Galbot/galbot_g1/galbot_g1.usda"
ROBOT_PRIM = "/World/Robot"

# Real box assets (Isaac's YCB set) instead of procedural cubes - both confirmed live to already
# carry RigidBodyAPI + MassAPI (see spawn_real_box, which overrides the mass to task-appropriate
# values but leaves collision/rigid-body setup as authored). Native footprints (x,y,z meters,
# measured via a live AABB probe at scale 1.0): cracker box ~(0.164, 0.213, 0.072), sugar box
# ~(0.093, 0.176, 0.045). BOX_ASSET_MAIN is reused (at a bigger scale) for the third/biggest box
# too - only two physics-ready box-shaped assets exist in this asset library version.
BOX_ASSET_MAIN = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
BOX_ASSET_EXTRA = "/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

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

# A second table, purely decorative - not a pick/place target, not touched by the recorder or
# the task/manifest. Placed far across the room from the table+cart cluster (which sits near the
# origin), well clear of ROOM_MIN/ROOM_MAX's walls.
SECOND_TABLE_POSITION = (8.0, -6.0)

# Gap (meters, edge to edge) between adjacent boxes when --extra-boxes lays out 3 side by side on
# the table. Not verified against the actual table asset's footprint (unknown until runtime) -
# if a box ends up hanging off the table edge, either shrink this, shrink --cube2-size/
# --cube3-size, or check the table asset's real width via the geometry diagnostic printout.
CUBE_ROW_GAP_M = 0.05

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

MAX_JOINT_LEAD_RAD = 0.3
ARM_CONTACT_MAX_LEAD_RAD = 0.1


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


def place_on_ground(bbox_cache, prim_path: str, x: float, y: float, scale: float = 1.0) -> np.ndarray:
    """Move a freshly-referenced (identity-transform) prim so its footprint is centered at
    (x, y) and its lowest point rests on z=0. Ported from
    ../Robot_project/capture_cube_rgbd.py - needed here (unlike stream_demo.py, which hardcodes
    the table's position) so the table/cart/robot cluster's exact AABBs are known and
    reproducible, which the cart-adjacency placement below depends on.
    """
    aabb0 = compute_world_aabb(bbox_cache, prim_path) * scale
    center_x0 = (aabb0[0] + aabb0[3]) / 2.0
    center_y0 = (aabb0[1] + aabb0[4]) / 2.0
    position = np.array([x - center_x0, y - center_y0, -aabb0[2]])
    SingleXFormPrim(prim_path, position=position, scale=np.array([scale, scale, scale]))
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


def spawn_real_box(
    bbox_cache, assets_root_path: str, usd_relpath: str, prim_path: str, x: float, y: float, surface_z: float, scale: float, mass: float
) -> np.ndarray:
    """Reference a physics-ready real box asset (see BOX_ASSET_MAIN/BOX_ASSET_EXTRA - both
    confirmed to already carry RigidBodyAPI + MassAPI) and place it resting on `surface_z`,
    overriding its authored mass to `mass` kg (the asset's own baked-in mass isn't necessarily
    tuned for this task's friction-only hug hold). Friction is left at the asset's own baked-in
    default for now - not yet worth the risk of mixing the isaacsim.core.api (legacy) and
    isaacsim.core.experimental physics-material APIs without live verification; if the hug hold
    proves unreliable, binding a custom high-friction PhysicsMaterial here is the next thing to
    try, not a kinematic attach (see the module docstring).
    """
    add_reference_to_stage(usd_path=assets_root_path + usd_relpath, prim_path=prim_path)
    aabb = place_on_surface(bbox_cache, prim_path, x=x, y=y, surface_z=surface_z, scale=scale)
    UsdPhysics.MassAPI(get_prim_at_path(prim_path)).GetMassAttr().Set(mass)
    return aabb


# Pushcart geometry - ported from ../Robot_project/capture_cube_rgbd.py's build_pushcart, with
# one change: a deck_riser_height parameter (see pushcart_deck_top_z / --deck-riser) inserted
# between the caster assembly and the deck, since the stock ~0.15m deck height was designed for
# "push by the handle," not "place a box here," and is likely well below table height.
PUSHCART_DECK_HALF_EXTENT = (0.3, 0.225)
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
    """Buffers one episode's RGB frames + proprioception/action vectors in memory (a few
    seconds at 15Hz/640x480 is a few hundred MB at most - fine to hold in RAM) and writes it to
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
        self.states: list = []
        self.actions: list = []

    def start(self) -> None:
        self._reset_buffer()

    def append(self, rgb: np.ndarray, state: np.ndarray, action: np.ndarray) -> None:
        self.frames.append(np.ascontiguousarray(rgb[:, :, :3]))
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
    bbox_cache = bounds_utils.create_bbox_cache()

    add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table")
    table_aabb = place_on_ground(bbox_cache, "/World/Table", x=0.0, y=0.0)
    table_center_x = (table_aabb[0] + table_aabb[3]) / 2.0
    table_center_y = (table_aabb[1] + table_aabb[4]) / 2.0
    table_top_z = table_aabb[5]

    # Second table - decorative only, not part of the recorded task (see SECOND_TABLE_POSITION).
    add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table2")
    place_on_ground(bbox_cache, "/World/Table2", x=SECOND_TABLE_POSITION[0], y=SECOND_TABLE_POSITION[1])

    # Cart placed beside the table, front edge flush with the table's own near (-x) edge, so
    # both sit at the same approach depth and only differ in y - see ROBOT_APPROACH_GAP_M's
    # comment above for why this exact layout is a first guess, not a verified one.
    dx, dy = PUSHCART_DECK_HALF_EXTENT
    cart_x = table_aabb[0] + dx
    cart_y = table_aabb[4] + CART_TABLE_GAP_M + dy
    build_pushcart(stage, "/World/PushCart", x=cart_x, y=cart_y, deck_riser_height=args.deck_riser)

    add_reference_to_stage(usd_path=assets_root_path + ROBOT_ASSET, prim_path=ROBOT_PRIM)
    robot_spawn_x = table_aabb[0] - ROBOT_APPROACH_GAP_M
    robot_spawn_y = (table_center_y + cart_y) / 2.0  # centered between table and cart
    place_on_ground(bbox_cache, ROBOT_PRIM, x=robot_spawn_x, y=robot_spawn_y)

    if args.cube_start == "table":
        main_box_aabb = spawn_real_box(
            bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
            x=table_center_x, y=table_center_y, surface_z=table_top_z, scale=args.cube_scale, mass=args.cube_mass,
        )
    else:
        cart_deck_top_z_for_spawn = pushcart_deck_top_z(args.deck_riser)
        main_box_aabb = spawn_real_box(
            bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
            x=cart_x, y=cart_y, surface_z=cart_deck_top_z_for_spawn, scale=args.cube_scale, mass=args.cube_mass,
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

        add_reference_to_stage(usd_path=assets_root_path + BOX_ASSET_EXTRA, prim_path="/World/Cube2")
        cube2_footprint = scaled_footprint(bbox_cache, "/World/Cube2", args.cube2_scale)
        cube2_half_dy = (cube2_footprint[4] - cube2_footprint[1]) / 2.0
        cube2_y = table_center_y + main_half_dy + CUBE_ROW_GAP_M + cube2_half_dy
        place_on_surface(bbox_cache, "/World/Cube2", x=table_center_x, y=cube2_y, surface_z=table_top_z, scale=args.cube2_scale)
        UsdPhysics.MassAPI(get_prim_at_path("/World/Cube2")).GetMassAttr().Set(args.cube2_mass)

        add_reference_to_stage(usd_path=assets_root_path + BOX_ASSET_MAIN, prim_path="/World/Cube3")
        cube3_footprint = scaled_footprint(bbox_cache, "/World/Cube3", args.cube3_scale)
        cube3_half_dy = (cube3_footprint[4] - cube3_footprint[1]) / 2.0
        cube3_y = table_center_y - main_half_dy - CUBE_ROW_GAP_M - cube3_half_dy
        place_on_surface(bbox_cache, "/World/Cube3", x=table_center_x, y=cube3_y, surface_z=table_top_z, scale=args.cube3_scale)
        UsdPhysics.MassAPI(get_prim_at_path("/World/Cube3")).GetMassAttr().Set(args.cube3_mass)

        print(
            f"[geometry] cube2 (sugar box) half_dy={cube2_half_dy:.3f}m at y={cube2_y:.3f}  "
            f"cube3 (cracker box) half_dy={cube3_half_dy:.3f}m at y={cube3_y:.3f}  "
            f"(check none hang off the table edge - Stage-panel + F on /World/Cube2, /World/Cube3 to verify)"
        )

    camera = Camera(
        prim_path=f"{ROBOT_PRIM}/OmniChassis/base_link/front_camera",
        frequency=20,
        resolution=(640, 480),
    )

    robot = SingleArticulation(ROBOT_PRIM)

    world.reset()
    robot.initialize()
    camera.initialize()

    aperture = camera.get_horizontal_aperture()
    camera.set_focal_length(float(aperture / (2.0 * np.tan(np.radians(60.0) / 2.0))))
    camera.set_local_pose(translation=np.array([0.4, 0.0, 0.9]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))

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
        camera_key="front_camera",
        image_hw=(480, 640),
        task_name=args.task,
    )
    recorder_state = RecorderState.IDLE

    cart_deck_top_z = pushcart_deck_top_z(args.deck_riser)
    print(
        f"[geometry] table_top_z={table_top_z:.3f}m  cart_deck_top_z={cart_deck_top_z:.3f}m  "
        f"delta={table_top_z - cart_deck_top_z:+.3f}m  cube_scale={args.cube_scale}  "
        f"(if delta is large and positive, raise --deck-riser; see module docstring's Stage 0 check)"
    )

    left_arm_swing_rate = arm_swing_rate("left", args.arm_speed)
    right_arm_swing_rate = arm_swing_rate("right", args.arm_speed)
    left_arm_swing_fraction = 0.0
    right_arm_swing_fraction = 0.0
    torso_height_fraction = 0.0
    hand_updown_rad = 0.0
    gripper_rad = 0.0

    held_keys: set = set()
    reset_requested = False
    space_requested = False
    label_success_requested = False
    label_fail_requested = False
    discard_requested = False

    def on_keyboard_event(event, *_args, **_kwargs) -> bool:
        nonlocal reset_requested, space_requested, label_success_requested, label_fail_requested, discard_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                reset_requested = True
            elif event.input == carb.input.KeyboardInput.SPACE:
                space_requested = True
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
            ):
                held_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            held_keys.discard(event.input)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("Controls: W/S drive forward/back, A/D strafe left/right, Q/E rotate, I/K torso up/down.")
    print("  Hold U: both arms swing forward (shoulder only). Hold O: swing back to open.")
    print("  Hold J: both hands raise (elbow only). Hold L: both hands lower.")
    print("  Hold M: both grippers close. Hold N: both grippers open.")
    print("  SPACE: start/stop episode recording. After stop: Y=success, F=failure, Backspace=discard.")
    print("  R resets (also discards an in-progress episode). Close the window to exit.")

    physics_dt = world.get_physics_dt()
    record_period = 1.0 / args.record_fps
    record_accum = 0.0

    while simulation_app.is_running():
        if reset_requested:
            reset_requested = False
            if recorder_state is not RecorderState.IDLE:
                print("Reset requested mid-episode - discarding the buffered episode.")
                recorder.discard()
                recorder_state = RecorderState.IDLE
            world.reset()
            robot.initialize()
            left_arm_swing_fraction = 0.0
            right_arm_swing_fraction = 0.0
            torso_height_fraction = 0.0
            hand_updown_rad = 0.0
            gripper_rad = 0.0
            record_accum = 0.0
            continue

        if space_requested:
            space_requested = False
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

        world.step(render=True)

        record_accum += physics_dt
        if recorder_state is RecorderState.RECORDING and record_accum >= record_period:
            record_accum -= record_period
            rgba = camera.get_rgba()
            if rgba is not None:
                state_vec = robot.get_joint_positions()[state_dof_indices].astype(np.float32)
                action_vec = np.concatenate([left_arm_q, right_arm_q, torso_q, left_gripper_q, right_gripper_q]).astype(
                    np.float32
                )
                recorder.append(rgba, state_vec, action_vec)

    simulation_app.close()


if __name__ == "__main__":
    main()
