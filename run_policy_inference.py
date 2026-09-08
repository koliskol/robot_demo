"""Live closed-loop testing of trained pickup/place policies in Isaac Sim - no recording, no
EpisodeRecorder, no B/Y/F/Backspace label workflow. This is collect_pickplace_demo.py's --rollout
mode with everything data-collection-specific stripped out, for the narrower job of just watching
a checkpoint drive the robot: build the same table(+table2/cart) scene, connect to two running
policy_server.py processes (pickup + place), and let M/N toggle which one (if any) is currently
driving the arms/torso/grippers/chassis-forward, while a human drives the rest of the chassis
manually to reposition between the two.

Usage - two policy_server.py processes (lerobot env) must already be running, one per checkpoint:

    conda run -n lerobot python policy_server.py --checkpoint-dir .../pickup_policy/checkpoints/last/pretrained_model \\
        --dataset-root ./lerobot_dataset_pickup --dataset-repo-id local/pickup_policy --port 8765
    conda run -n lerobot python policy_server.py --checkpoint-dir .../place_policy/checkpoints/last/pretrained_model \\
        --dataset-root ./lerobot_dataset_place --dataset-repo-id local/place_policy --port 8766

Then, in the isaac_sim env:

    conda run -n isaac_sim python run_policy_inference.py

Controls (viewport window must have focus):

    W / S       drive forward / backward - your key press always wins the instant you touch either
                one. While you're not touching them, an active policy's own chassis_forward
                prediction drives instead (that's part of the trained action space); with no
                policy active it just sits idle. Either way the policy always receives an accurate
                chassis_forward observation, whether or not its own output gets applied.
    A / D       strafe left / right - always manual, never part of the trained action space.
    Q / E       rotate left / right - always manual, same reason.
    M           toggle the PICKUP policy on/off - press once to activate (arms/torso/grippers
                become policy-controlled, forward-drive too whenever you're not on W/S), press
                again to deactivate (freezes the current pose - does NOT snap back to idle, so it
                won't drop anything being held).
    N           same toggle for the PLACE policy. Activating one deactivates the other implicitly
                (only one policy drives the robot at a time) - just press M or N again, no need to
                turn the current one off first.
    R           reset the robot/box/table2 to a fresh (jittered) spawn pose, deactivating whatever
                policy was active.

    Close the viewport window to exit.

This is deliberately NOT collect_pickplace_demo.py --rollout with flags to disable recording -
that script's controls (B to start/stop, Y/F/Backspace to label) exist because its job is
producing labeled training episodes; this script's only job is "watch the policy drive the robot
right now," so those don't apply and having to press B before every attempt was just friction (see
CLAUDE.md/session history for the current script's own dedicated-episode workflow if you actually
want to add to raw_episodes/rollout_episodes - that's still collect_pickplace_demo.py's job, not
this one's).

Scene-building, physics, and camera code below is copied verbatim from collect_pickplace_demo.py
(same Galbot G1 asset, same table/box/pushcart/table2 setup, same head-mounted camera derivation -
see that script's own module docstring and CLAUDE.md's "LeRobot pick-and-place pipeline" section
for the full derivation of all of it) - only the control loop and argparse surface differ.
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--host", type=str, default="0.0.0.0", help="WebRTC viewing server bind address.")
parser.add_argument("--port", type=int, default=8080, help="WebRTC viewing server port.")
parser.add_argument("--policy-fps", type=float, default=15.0, help="Rate to query the active policy at - match --record-fps of the data the checkpoints were trained on.")
parser.add_argument("--policy-host", type=str, default="127.0.0.1", help="policy_server.py host for the pickup policy (M key).")
parser.add_argument("--policy-port", type=int, default=8765, help="policy_server.py port for the pickup policy (M key).")
parser.add_argument("--policy2-host", type=str, default="127.0.0.1", help="policy_server.py host for the place policy (N key).")
parser.add_argument("--policy2-port", type=int, default=8766, help="policy_server.py port for the place policy (N key).")
parser.add_argument("--task", type=str, default="pickup_policy", help="Task label sent to the pickup policy (M key) - must match what its checkpoint was trained on.")
parser.add_argument("--task2", type=str, default="place_policy", help="Task label sent to the place policy (N key) - must match what its checkpoint was trained on.")
parser.add_argument(
    "--place-target",
    type=str,
    choices=["cart", "table2"],
    default="table2",
    help="Which secondary object the scene builds as the table's pick/place partner - defaults to "
    "table2 here (unlike collect_pickplace_demo.py's cart default) since that's what the current "
    "trained checkpoints expect. Must match the scene the checkpoints were trained on.",
)
parser.add_argument("--table2-jitter-m", type=float, default=0.05, help="Max random xy offset (meters) applied to table2's spawn/reset position (--place-target=table2 only).")
parser.add_argument("--table2-yaw-jitter-deg", type=float, default=5.0, help="Max random yaw offset (degrees) applied to table2's spawn/reset orientation.")
parser.add_argument(
    "--cube-start",
    type=str,
    choices=["table", "cart", "table2"],
    default="table",
    help="Where the cube spawns on reset - must be 'table' or match --place-target, and should "
    "match whichever policy you're about to activate first (pickup expects 'table').",
)
parser.add_argument("--cube-scale", type=float, default=1.0, help="Uniform scale multiplier for the main box.")
parser.add_argument("--cube-mass", type=float, default=0.15, help="Main box mass in kg.")
parser.add_argument("--box-jitter-m", type=float, default=0.03, help="Max random xy offset (meters) applied to the main box's spawn/reset position.")
parser.add_argument("--box-yaw-jitter-deg", type=float, default=10.0, help="Max random yaw offset (degrees) applied to the main box's spawn/reset orientation.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the box/table2 spawn-jitter RNG. Default: a fresh random sequence each run.")
parser.add_argument("--cube2-scale", type=float, default=1.0, help="Scale multiplier for a second, bigger distractor box (table-start only).")
parser.add_argument("--cube2-mass", type=float, default=0.25, help="Mass of the second box in kg.")
parser.add_argument("--cube3-scale", type=float, default=1.0, help="Scale multiplier for a third, even bigger distractor box (table-start only).")
parser.add_argument("--cube3-mass", type=float, default=0.35, help="Mass of the third box in kg.")
parser.add_argument("--extra-boxes", dest="extra_boxes", action="store_true", default=True, help="Spawn the two extra distractor boxes (default: on, table-start only).")
parser.add_argument("--no-extra-boxes", dest="extra_boxes", action="store_false", help="Spawn only the single primary box.")
parser.add_argument("--table-height-scale", type=float, default=0.69, help="Height-only scale factor for the main table.")
parser.add_argument("--deck-riser", type=float, default=0.0, help="Extra meters added to the pushcart deck's stock height (--place-target=cart only).")
parser.add_argument("--drive-speed", type=float, default=0.4, help="Chassis drive/strafe command magnitude (manual W/A/S/D, and caps the policy's own forward/back command).")
parser.add_argument("--turn-speed", type=float, default=0.15, help="Chassis rotation command magnitude (manual Q/E).")
args = parser.parse_args()
if args.cube_start not in ("table", args.place_target):
    parser.error(f"--cube-start {args.cube_start!r} requires --place-target {args.cube_start!r} (got --place-target {args.place_target!r})")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow
import omni.graph.core as og
import isaacsim.core.utils.bounds as bounds_utils
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
from policy_client import PolicyClient

TABLE_ASSET = "/Isaac/Environments/Office/Props/SM_TableB.usd"
ROBOT_ASSET = "/Isaac/Robots/Galbot/galbot_g1/galbot_g1.usda"
ROBOT_PRIM = "/World/Robot"

# Camera is head-mounted - see collect_pickplace_demo.py's HEAD_CAMERA_MOUNT comment for the full
# derivation (identical asset/mount here, copied verbatim).
HEAD_CAMERA_MOUNT = (
    f"{ROBOT_PRIM}/OmniChassis/base_link/omni_chassis_base_link/omni_chassis_leg_mount_link/leg_base_link/"
    "leg_link1/leg_link2/leg_link3/leg_link4/leg_link5/leg_end_effector_mount_link/torso_base_link/Head_Golf/"
    "torso_base_link/torso_head_mount_link/head_base_link/head_link1/head_link2/head_end_effector_mount_link"
)

HEAD_JOINT_PATHS = [
    f"{ROBOT_PRIM}/OmniChassis/base_link/omni_chassis_base_link/omni_chassis_leg_mount_link/leg_base_link/"
    f"leg_link1/leg_link2/leg_link3/leg_link4/leg_link5/leg_end_effector_mount_link/torso_base_link/Head_Golf/"
    f"joints/head_joint{i}"
    for i in (1, 2)
]


def stiffen_head_joints() -> None:
    """See collect_pickplace_demo.py's stiffen_head_joints - identical, copied verbatim."""
    for path in HEAD_JOINT_PATHS:
        drive = UsdPhysics.DriveAPI.Get(get_prim_at_path(path), "angular")
        drive.GetStiffnessAttr().Set(200.0)
        drive.GetDampingAttr().Set(20.0)


def hold_head_joints(robot: SingleArticulation, head_dof_indices: list) -> None:
    robot.apply_action(ArticulationAction(joint_positions=np.zeros(2), joint_indices=head_dof_indices))


CAMERA_ROLL_DEG = 90.0
CAMERA_TILT_DEG = 26.7
CAMERA_FOV_DEG = 90.0
CAMERA_PAN_DEG = 22.0
CAMERA_MOUNT_FORWARD_OFFSET_M = 0.1
CAMERA_NEAR_CLIP_M = 0.1
CAMERA_FAR_CLIP_M = 50.0


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    roll_half = np.radians(roll_deg) / 2.0
    roll_q = np.array([np.cos(roll_half), np.sin(roll_half), 0.0, 0.0])
    tilt_half = np.radians(tilt_deg) / 2.0
    tilt_q = np.array([np.cos(tilt_half), 0.0, np.sin(tilt_half), 0.0])
    return quat_multiply(tilt_q, roll_q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


CAMERA_WU_QUAT = np.array([0.5, 0.5, -0.5, -0.5])


def camera_pan_tilt_quat(roll_deg: float, tilt_deg: float, pan_deg: float, mount_world_quat: np.ndarray) -> np.ndarray:
    q_base = camera_head_mount_quat(roll_deg, tilt_deg)
    q_cam_world = quat_multiply(quat_multiply(mount_world_quat, q_base), CAMERA_WU_QUAT)
    pan_half = np.radians(pan_deg) / 2.0
    q_yaw = np.array([np.cos(pan_half), 0.0, 0.0, np.sin(pan_half)])
    q_new_world = quat_multiply(q_yaw, q_cam_world)
    q_new_local = quat_multiply(quat_multiply(quat_conjugate(mount_world_quat), q_new_world), quat_conjugate(CAMERA_WU_QUAT))
    return q_new_local / np.linalg.norm(q_new_local)


BOX_ASSET_MAIN = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd"
BOX_ASSET_CUBE2 = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd"
BOX_ASSET_CUBE3 = "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd"

ROOM_MIN = (-6.0, -8.0)
ROOM_MAX = (12.0, 10.0)
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1

ROBOT_APPROACH_GAP_M = 0.9
CART_TABLE_GAP_M = 0.15

TABLE2_GAP_M = 1.2
TABLE2_EDGE_INSET_M = 0.3
TABLE2_SIDE_SIGN = 1.0

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
    w, x, y, z = orientation_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


ROBOT_FORWARD_OFFSET_RAD = np.pi / 2.0


def robot_forward_reference(robot: SingleArticulation) -> tuple:
    """(position_xy, forward_unit_vector) for the chassis_forward state dim - call whenever a
    policy is (re)activated, so forward_displacement_m is measured from that point on."""
    position, orientation_wxyz = robot.get_world_pose()
    heading = robot_heading_yaw(orientation_wxyz) - ROBOT_FORWARD_OFFSET_RAD
    return np.array(position[:2]), np.array([np.cos(heading), np.sin(heading)])


def forward_displacement(robot: SingleArticulation, start_pos_xy: np.ndarray, forward_dir: np.ndarray) -> float:
    position, _ = robot.get_world_pose()
    return float(np.dot(np.array(position[:2]) - start_pos_xy, forward_dir))


# Idle/fallback pose while no policy is active - matches collect_pickplace_demo.py's
# STARTING_*-fraction pose exactly (see that file's comment for the derivation), NOT the fully-open
# rest pose.
ARM_FORWARD_POSE = {
    "left": [0.0, -1.308997, 0.0, 0.0, 0.0, 0.0, 0.0],
    "right": [0.0, -1.608100, 0.0, 0.0, 0.0, 0.0, 0.0],
}
ARM_OPEN_POSE = {
    "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
ARM_HAND_UPDOWN_JOINT_INDEX = 3
STARTING_LEFT_ARM_SWING_FRACTION = 0.815
STARTING_RIGHT_ARM_SWING_FRACTION = 0.663
STARTING_HAND_UPDOWN_RAD = 1.173
TORSO_UP_POSE = [0.0, 0.0, 0.0, 0.0, 0.0]

MAX_JOINT_LEAD_RAD = 0.3
ARM_CONTACT_MAX_LEAD_RAD = 0.03
GRIPPER_MAX_LEAD_RAD = 0.008


def idle_arm_pose() -> tuple:
    """The settle-into pose used only at launch/reset, before any policy has ever activated."""
    left = (1.0 - STARTING_LEFT_ARM_SWING_FRACTION) * np.array(ARM_OPEN_POSE["left"]) + STARTING_LEFT_ARM_SWING_FRACTION * np.array(
        ARM_FORWARD_POSE["left"]
    )
    right = (1.0 - STARTING_RIGHT_ARM_SWING_FRACTION) * np.array(
        ARM_OPEN_POSE["right"]
    ) + STARTING_RIGHT_ARM_SWING_FRACTION * np.array(ARM_FORWARD_POSE["right"])
    left[ARM_HAND_UPDOWN_JOINT_INDEX] += STARTING_HAND_UPDOWN_RAD
    right[ARM_HAND_UPDOWN_JOINT_INDEX] += STARTING_HAND_UPDOWN_RAD
    return left, right

# Camera pan/tilt calibration keys (arrow keys).
CAMERA_ROTATE_KEY_SPEED_DEG_S = 20.0
CAMERA_ROTATE_STEP_DEG = 2.0
CAMERA_ROTATE_KEYS_PAN = {
    carb.input.KeyboardInput.LEFT: 1.0,
    carb.input.KeyboardInput.RIGHT: -1.0,
}
CAMERA_ROTATE_KEYS_TILT = {
    carb.input.KeyboardInput.DOWN: 1.0,
    carb.input.KeyboardInput.UP: -1.0,
}


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
    r = jitter_m * np.sqrt(rng.uniform(0.0, 1.0))
    theta = rng.uniform(0.0, 2.0 * np.pi)
    dx, dy = r * np.cos(theta), r * np.sin(theta)
    yaw_deg = rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)
    yaw_half = np.radians(yaw_deg) / 2.0
    yaw_quat = np.array([np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)])
    return dx, dy, yaw_deg, yaw_quat


def place_on_ground(bbox_cache, prim_path: str, x: float, y: float, scale: float = 1.0, z_scale: float = None) -> np.ndarray:
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
    aabb0 = compute_world_aabb(bbox_cache, prim_path) * scale
    center_x0 = (aabb0[0] + aabb0[3]) / 2.0
    center_y0 = (aabb0[1] + aabb0[4]) / 2.0
    position = np.array([x - center_x0, y - center_y0, surface_z - aabb0[2]])
    SingleXFormPrim(prim_path, position=position, scale=np.array([scale, scale, scale]))
    bbox_cache.Clear()
    return compute_world_aabb(bbox_cache, prim_path)


def scaled_footprint(bbox_cache, prim_path: str, scale: float) -> np.ndarray:
    return compute_world_aabb(bbox_cache, prim_path) * scale


def make_box_dynamic(prim_path: str, mass: float) -> None:
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
    add_reference_to_stage(usd_path=assets_root_path + usd_relpath, prim_path=prim_path)
    make_box_dynamic(prim_path, mass)
    return place_on_surface(bbox_cache, prim_path, x=x, y=y, surface_z=surface_z, scale=scale)


PUSHCART_DECK_HALF_EXTENT = (0.45, 0.225)
PUSHCART_DECK_THICKNESS = 0.03
PUSHCART_WHEEL_RADIUS = 0.05
PUSHCART_HANDLE_POST_HEIGHT = 0.75
PUSHCART_CHASSIS_MASS = 4.4
PUSHCART_FORK_MASS = 0.05
PUSHCART_WHEEL_MASS = 0.1
CASTER_ROLLING_FRICTION_NM = 0.05


def pushcart_deck_top_z(deck_riser_height: float) -> float:
    deck_bottom_z = 2.0 * PUSHCART_WHEEL_RADIUS + 0.02 + deck_riser_height
    return deck_bottom_z + PUSHCART_DECK_THICKNESS


def build_pushcart(stage, prim_path: str, x: float, y: float, deck_riser_height: float = 0.0) -> None:
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


def main() -> None:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not resolve the Isaac Sim assets root path (check network access).")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = get_current_stage()

    build_room(stage)

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

    rng = np.random.default_rng(args.seed)

    table2_xform = None
    table2_anchor_x = table2_anchor_y = table2_anchor_z = None
    if args.place_target == "cart":
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
    stiffen_head_joints()
    robot_spawn_x = table_aabb[0] - ROBOT_APPROACH_GAP_M
    robot_spawn_y = (table_center_y + target_y) / 2.0
    place_on_ground(bbox_cache, ROBOT_PRIM, x=robot_spawn_x, y=robot_spawn_y)

    box_center_x, box_center_y = (table_center_x, table_center_y) if args.cube_start == "table" else (target_x, target_y)
    box_surface_z = table_top_z if args.cube_start == "table" else target_top_z
    box_dx, box_dy, box_yaw_deg, box_yaw_quat = sample_pose_jitter(rng, args.box_jitter_m, args.box_yaw_jitter_deg)
    main_box_aabb = spawn_real_box(
        bbox_cache, assets_root_path, BOX_ASSET_MAIN, "/World/Cube",
        x=box_center_x + box_dx, y=box_center_y + box_dy, surface_z=box_surface_z, scale=args.cube_scale, mass=args.cube_mass,
    )
    box_xform = SingleXFormPrim("/World/Cube")
    box_xform.set_world_pose(orientation=box_yaw_quat)
    box_anchor_z = float(box_xform.get_world_pose()[0][2])
    print(
        f"[box] initial spawn offset dx={box_dx:+.3f}m dy={box_dy:+.3f}m yaw={box_yaw_deg:+.1f}deg "
        f"(--box-jitter-m={args.box_jitter_m} --box-yaw-jitter-deg={args.box_yaw_jitter_deg})"
    )

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
    # 22-dim: 21 joints + chassis_forward appended separately - must match what the checkpoints
    # were trained on (see CLAUDE.md's "recorded state/action space grew from 21 to 22 dims").

    print(
        f"[geometry] table_top_z={table_top_z:.3f}m  {args.place_target}_top_z={target_top_z:.3f}m  "
        f"delta={table_top_z - target_top_z:+.3f}m  cube_scale={args.cube_scale}"
    )

    policy_client_pickup = PolicyClient(args.policy_host, args.policy_port)
    policy_client_pickup.connect()
    print(f"[inference] connected to pickup policy_server.py at {args.policy_host}:{args.policy_port} (task={args.task!r}, M to toggle)")
    policy_client_place = PolicyClient(args.policy2_host, args.policy2_port)
    policy_client_place.connect()
    print(f"[inference] connected to place policy_server.py at {args.policy2_host}:{args.policy2_port} (task={args.task2!r}, N to toggle)")

    active_policy_client = None  # None until M (pickup) or N (place) toggles one on
    active_task = None
    policy_action_vec = None  # last action received from the active policy; None whenever no policy is driving
    episode_start_pos_xy = None
    episode_forward_dir = None

    # Arm/torso/gripper targets persist across frames and are only updated while a policy is
    # active (see the main loop below) - deactivating a policy (second M/N press) now HOLDS
    # whatever pose the arms were last in, rather than snapping back to the idle/open pose. The
    # earlier version reset straight to idle on deactivate, which yanked the arms open and dropped
    # anything being held (e.g. mid-hug) - freezing in place avoids that. Only used as the actual
    # starting values here, at launch, before any policy has ever activated.
    held_left_arm_q, held_right_arm_q = idle_arm_pose()
    held_torso_q = np.array(TORSO_UP_POSE)
    held_left_gripper_target = np.array([0.0])
    held_right_gripper_target = np.array([0.0])

    frame_store = FrameStore()
    run_in_background(
        frame_store, host=args.host, port=args.port, static_index="collect_index.html", static_viewer_js="collect_viewer.js"
    )

    held_keys: set = set()
    reset_requested = False
    pickup_toggle_requested = False
    place_toggle_requested = False
    camera_print_requested = False

    def on_keyboard_event(event, *_args, **_kwargs) -> bool:
        nonlocal reset_requested, pickup_toggle_requested, place_toggle_requested, camera_print_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                reset_requested = True
            elif event.input == carb.input.KeyboardInput.M:
                pickup_toggle_requested = True
            elif event.input == carb.input.KeyboardInput.N:
                place_toggle_requested = True
            elif event.input in DRIVE_KEY_AXES or event.input in CAMERA_ROTATE_KEYS_PAN or event.input in CAMERA_ROTATE_KEYS_TILT:
                held_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in CAMERA_ROTATE_KEYS_PAN or event.input in CAMERA_ROTATE_KEYS_TILT:
                camera_print_requested = True
            held_keys.discard(event.input)
        return True

    input_interface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    print("Controls: W/S drive forward/back (your key press always wins - policy drives forward/back")
    print("  only while you're not touching W/S), A/D strafe, Q/E rotate (always manual).")
    print(f"  M: toggle PICKUP policy on/off (task={args.task!r}).  N: toggle PLACE policy on/off (task={args.task2!r}).")
    print("  Only one policy drives at a time - activating one implicitly deactivates the other.")
    print("  Arrow keys: rotate the camera (Left/Right pan, Up/Down tilt). Prints pan/tilt on release.")
    print("  R resets the scene (also deactivates the current policy). Close the window to exit.")

    physics_dt = world.get_physics_dt()
    query_period = 1.0 / args.policy_fps
    query_accum = 0.0
    consecutive_none_joint_frames = 0
    physics_hz = 1.0 / physics_dt
    debug_frame_counter = 0
    reset_count = 0

    while simulation_app.is_running():
        debug_frame_counter += 1
        if debug_frame_counter % 60 == 0:
            active_name = "PICKUP" if active_policy_client is policy_client_pickup else "PLACE" if active_policy_client is policy_client_place else None
            print(f"[debug] resets={reset_count} held_keys={sorted(k.name for k in held_keys)} active_policy={active_name} policy_action_vec_is_none={policy_action_vec is None}")
        for cmd in frame_store.pop_commands():
            if cmd.get("action") == "camera_rotate":
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
            reset_count += 1
            active_policy_client = None
            active_task = None
            policy_action_vec = None
            print("Reset requested - deactivating the active policy (if any).")
            world.reset()
            robot.initialize()
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
                print(f"[table2] spawn offset dx={table2_dx:+.3f}m dy={table2_dy:+.3f}m yaw={table2_yaw_deg:+.1f}deg")
            box_dx, box_dy, box_yaw_deg, box_yaw_quat = sample_pose_jitter(rng, args.box_jitter_m, args.box_yaw_jitter_deg)
            box_xform.set_world_pose(
                position=np.array([box_center_x + box_dx, box_center_y + box_dy, box_anchor_z]),
                orientation=box_yaw_quat,
            )
            print(f"[box] spawn offset dx={box_dx:+.3f}m dy={box_dy:+.3f}m yaw={box_yaw_deg:+.1f}deg")
            held_left_arm_q, held_right_arm_q = idle_arm_pose()
            held_torso_q = np.array(TORSO_UP_POSE)
            held_left_gripper_target = np.array([0.0])
            held_right_gripper_target = np.array([0.0])
            query_accum = 0.0
            episode_start_pos_xy = None
            episode_forward_dir = None
            continue

        if pickup_toggle_requested:
            pickup_toggle_requested = False
            if active_policy_client is policy_client_pickup:
                active_policy_client = None
                active_task = None
                policy_action_vec = None
                print("[inference] PICKUP deactivated")
            else:
                active_policy_client = policy_client_pickup
                active_task = args.task
                active_policy_client.reset()
                policy_action_vec = None
                episode_start_pos_xy, episode_forward_dir = robot_forward_reference(robot)
                print(f"[inference] PICKUP activated (task={active_task!r})")

        if place_toggle_requested:
            place_toggle_requested = False
            if active_policy_client is policy_client_place:
                active_policy_client = None
                active_task = None
                policy_action_vec = None
                print("[inference] PLACE deactivated")
            else:
                active_policy_client = policy_client_place
                active_task = args.task2
                active_policy_client.reset()
                policy_action_vec = None
                episode_start_pos_xy, episode_forward_dir = robot_forward_reference(robot)
                print(f"[inference] PLACE activated (task={active_task!r})")

        # Repurposes collect_index.html/collect_viewer.js's recorder badge as a plain "policy
        # active?" indicator - RECORDING style whenever a policy is driving, IDLE otherwise. That
        # page's Start/Stop/label buttons are inert here (no command handler above reacts to them).
        frame_store.update_status(
            {"state": "RECORDING" if active_policy_client is not None else "IDLE", "episode_index": 0, "num_frames": 0}
        )

        # Chassis forward/back: your own W/S always wins the instant you press either one: while
        # neither is held, an active policy's own chassis_forward prediction drives instead (it
        # always receives an accurate chassis_forward *observation* either way, in the query_accum
        # block below - only whether its forward/back *action* output gets applied depends on this).
        # Strafe/rotate (A/D/Q/E) are always manual - never part of the trained action space.
        user_driving_forward = bool(held_keys & {carb.input.KeyboardInput.W, carb.input.KeyboardInput.S})
        command = compute_drive_command(held_keys, args.drive_speed, args.turn_speed)
        if policy_action_vec is not None and not user_driving_forward:
            command[0] = float(policy_action_vec[21])
        action = drive_controller.forward(command)
        if debug_frame_counter % 60 == 0 and any(command):
            print(f"[debug] command={command} wheel_velocities={list(action.joint_velocities)}")
        robot.apply_action(ArticulationAction(joint_velocities=action.joint_velocities, joint_indices=wheel_dof_indices))

        actual_q = robot.get_joint_positions()
        if actual_q is None:
            world.step(render=True)
            consecutive_none_joint_frames += 1
            if consecutive_none_joint_frames > 5 * physics_hz:
                print("[error] physics simulation view has not recovered in 5s - exiting.")
                break
            continue
        consecutive_none_joint_frames = 0

        # held_* only gets overwritten while a policy is actively driving - deactivating (M/N
        # again) simply stops updating it, freezing the arms/torso/grippers wherever they were
        # (see held_* init above for why: snapping to idle on deactivate used to drop the box).
        if policy_action_vec is not None:
            held_left_arm_q = policy_action_vec[0:7].copy()
            held_right_arm_q = policy_action_vec[7:14].copy()
            held_torso_q = policy_action_vec[14:19].copy()
            held_left_gripper_target = policy_action_vec[19:20].copy()
            held_right_gripper_target = policy_action_vec[20:21].copy()

        left_arm_q = clamp_to_actual(held_left_arm_q, actual_q[left_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        right_arm_q = clamp_to_actual(held_right_arm_q, actual_q[right_arm_dof_indices], max_lead=ARM_CONTACT_MAX_LEAD_RAD)
        robot.apply_action(ArticulationAction(joint_positions=left_arm_q, joint_indices=left_arm_dof_indices))
        robot.apply_action(ArticulationAction(joint_positions=right_arm_q, joint_indices=right_arm_dof_indices))

        torso_q = clamp_to_actual(held_torso_q, actual_q[leg_indices])
        robot.apply_action(ArticulationAction(joint_positions=torso_q, joint_indices=leg_indices))

        left_gripper_q = clamp_to_actual(held_left_gripper_target, actual_q[left_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
        right_gripper_q = clamp_to_actual(held_right_gripper_target, actual_q[right_gripper_dof_indices], max_lead=GRIPPER_MAX_LEAD_RAD)
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
            set_camera_orientation(camera_pan_deg, camera_tilt_deg)

        hold_head_joints(robot, head_dof_indices)

        world.step(render=True)

        rgba = camera.get_rgba()
        if rgba is not None:
            frame_store.update_rgb(rgba)
        depth = camera.get_depth()
        if depth is not None:
            frame_store.update_depth(depth)

        query_accum += physics_dt
        if active_policy_client is not None and query_accum >= query_period:
            query_accum -= query_period
            if rgba is not None:
                chassis_forward_state = forward_displacement(robot, episode_start_pos_xy, episode_forward_dir)
                state_vec = np.concatenate(
                    [robot.get_joint_positions()[state_dof_indices], [chassis_forward_state]]
                ).astype(np.float32)
                # Observation-to-action edge of the loop: the frame/state captured just above
                # becomes the target used by the joint-control block above on the *next* iteration
                # (one physics step of latency, negligible at 60Hz).
                policy_action_vec = active_policy_client.predict(rgba[:, :, :3], state_vec, active_task)

    policy_client_pickup.close()
    policy_client_place.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
