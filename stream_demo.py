"""Drive a Galbot G1 mobile robot around a small scene in Isaac Sim, while streaming its
chassis camera (RGB + normalized depth) and chassis lidar (point cloud) live to a browser
over WebRTC - see streaming_server.py for the server side and static/ for the web viewer.

Usage:

    conda run -n isaac_sim python stream_demo.py
    conda run -n isaac_sim python stream_demo.py --port 8080

Then open http://<this machine's address>:8080/ in a browser and click Connect.

Controls (viewport window must have focus):
    W / S       drive forward / backward
    A / D       strafe left / right
    Q / E       rotate left / right
    I / K       hold to move torso up / down (leg lift joints)
    U / O       hold to swing both arms forward / back out to open - shoulder joint only,
                rest of each arm (elbow/wrist/hand) stays straight and locked, never folds.
    J / L       hold to raise / lower both hands - elbow joint only, independent of U/O.
    M / N       hold to close / open both grippers.
    R           reset the robot to its spawn pose
    Close the viewport window to exit.

Camera and lidar mounting, and the arm/hand/torso jog controls (U/O/I/K/J/L/M/N) above, mirror
the sibling ../Robot_project/capture_cube_rgbd.py exactly (same Galbot G1 asset, same joint
targets/clamps/rates) - see that project's module docstring for the full derivation of the
pose constants and lead-clamping below (FK sweeps, live-tuned instability fixes, etc.); this
project deliberately still leaves out its pushcart/cube/capture-to-disk features to stay
focused on the streaming pipeline.
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--host", type=str, default="0.0.0.0", help="Streaming server bind address.")
parser.add_argument("--port", type=int, default=8080, help="Streaming server port.")
parser.add_argument("--drive-speed", type=float, default=1.0, help="Chassis drive/strafe command magnitude.")
parser.add_argument("--turn-speed", type=float, default=0.3, help="Chassis rotation command magnitude.")
parser.add_argument("--arm-speed", type=float, default=0.4, help="Max arm joint speed in radians/second.")
parser.add_argument("--torso-speed", type=float, default=0.4, help="Torso up/down speed, as a fraction/second of its full travel.")
args = parser.parse_args()

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
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

from streaming_server import FrameStore, run_in_background

TABLE_ASSET = "/Isaac/Environments/Office/Props/SM_TableB.usd"
ROBOT_ASSET = "/Isaac/Robots/Galbot/galbot_g1/galbot_g1.usda"
ROBOT_PRIM = "/World/Robot"
LIDAR_MOUNT_PATH = f"{ROBOT_PRIM}/OmniChassis/base_link/omni_chassis_base_link/chassis_lidar"

# Room footprint (meters, world xy) - sized to comfortably contain build_scene's table + 3
# landmark cubes below, with margin to drive around in. Also sent to the browser as-is (see
# FrameStore.update_world_state) so the map view (static/viewer.js) can draw the same outline.
ROOM_MIN = (-6.0, -8.0)
ROOM_MAX = (12.0, 10.0)
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.1

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


# Arm/hand/torso jog controls - ported as-is from the sibling ../Robot_project/
# capture_cube_rgbd.py (same asset, same joints); see that file's module docstring and the
# comments above each of these constants there for the full derivation (FK sweeps against
# generated URDFs, live-tuned instability fixes, etc.). Kept identical here rather than
# re-derived, since it's the same Galbot G1 asset and joint set.

# ARM_FORWARD_POSE / ARM_OPEN_POSE agree on every joint except each arm's shoulder
# (joint2), so U/O only ever swings the shoulder - elbow/wrist/hand stay locked straight.
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

# J/L jog both hands up/down via joint4 (the elbow), independent of U/O's shoulder swing.
# Additive on top of whichever swing pose is active (both poses leave joint4 at 0).
ARM_HAND_UPDOWN_JOINT_INDEX = 3  # joint4, 0-indexed into the 7-joint [joint1..joint7] chain
ARM_HAND_DOWN_MAX_RAD = 0.6
ARM_HAND_UP_MAX_RAD = 2.5  # joint4's actual upper limit is 2.568 rad on both arms
HAND_UPDOWN_KEYS = {
    carb.input.KeyboardInput.L: -1.0,  # toward hand down
    carb.input.KeyboardInput.J: 1.0,  # toward hand up
}

# M/N jog both hands' grippers open/closed (one drive joint per hand; five PhysxMimicJoint
# followers on each track it automatically). 0 = open, GRIPPER_CLOSE_MAX_RAD = closed.
GRIPPER_CLOSE_MAX_RAD = np.radians(97.57470703125)  # this drive joint's actual upper limit, both hands
GRIPPER_SPEED_RAD_S = 2.5
GRIPPER_KEYS = {
    carb.input.KeyboardInput.N: -1.0,  # toward open
    carb.input.KeyboardInput.M: 1.0,  # toward closed
}
# Tight lead clamp for the gripper specifically - closing on something solid means the fingers
# are *expected* to hit resistance and stop; a looser lead lets the position drive keep shoving
# into that contact, which live-tested unstable (whole arm flinging / gripper mechanism tearing
# loose). See capture_cube_rgbd.py's GRIPPER_MAX_LEAD_RAD comment for the full empirical sweep.
GRIPPER_MAX_LEAD_RAD = 0.008

# Torso lift: leg_joint1-3 are range-limited to [0, some positive max], 0 = straight/tall.
TORSO_UP_POSE = [0.0, 0.0, 0.0, 0.0, 0.0]
TORSO_DOWN_POSE = [0.8, 2.3, 1.55, 0.0, 0.0]
TORSO_HEIGHT_KEYS = {
    carb.input.KeyboardInput.I: -1.0,  # toward TORSO_UP_POSE
    carb.input.KeyboardInput.K: 1.0,  # toward TORSO_DOWN_POSE
}

# Max radians a jogged position command is allowed to run ahead of the joint's actual physical
# position, every frame (see clamp_to_actual) - keeps a held jog key from building unbounded
# PD error, and the resulting instability, once something physically blocks the joint.
MAX_JOINT_LEAD_RAD = 0.3
# Tighter lead for the arm (U/O's shoulder, J/L's elbow) specifically - MAX_JOINT_LEAD_RAD
# alone wasn't tight enough for "pushed into something rigid" (live-observed: joint velocities
# spiking, robot flung out of the scene). See capture_cube_rgbd.py's own comment for detail.
ARM_CONTACT_MAX_LEAD_RAD = 0.1


def arm_swing_rate(side: str, arm_speed: float) -> float:
    """Fraction/second rate for one arm's ARM_SWING_KEYS jog, chosen so its single moving
    joint (joint2) covers its OPEN-to-FORWARD distance at exactly `arm_speed` rad/second."""
    delta = abs(np.array(ARM_FORWARD_POSE[side]) - np.array(ARM_OPEN_POSE[side])).max()
    return arm_speed / float(delta)


def clamp_to_actual(target: np.ndarray, actual: np.ndarray, max_lead: float = MAX_JOINT_LEAD_RAD) -> np.ndarray:
    """Clamp a jogged position command so it never runs more than `max_lead` radians ahead of
    where that joint actually physically is - see MAX_JOINT_LEAD_RAD's comment above."""
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
    # Same rationale as the sibling project: these graphs depend on isaacsim.ros2.bridge,
    # which isn't loaded here (this demo always drives wheels directly, see build_pushcart's
    # sibling comment in ../Robot_project/capture_cube_rgbd.py for the full story), and left
    # in place they crash the moment they're ticked.
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
        linear_gain=-1.0,  # sign confirmed empirically in the sibling project
    )


def build_room(stage) -> None:
    """Four static (CollisionAPI only, no RigidBodyAPI - a fixed, never-moving collider,
    same as how the ground plane behaves) walls enclosing ROOM_MIN/ROOM_MAX. Walls overlap
    slightly at the corners (each wall's own half-extent is padded by WALL_THICKNESS along its
    length) so there's no gap for the lidar to see through or the robot to nose into.
    """
    (x0, y0), (x1, y1) = ROOM_MIN, ROOM_MAX
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width, depth = x1 - x0, y1 - y0

    # (center_x, center_y, half_extent_x, half_extent_y)
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


def robot_heading_yaw(orientation_wxyz: np.ndarray) -> float:
    """Yaw (rotation about Z, radians) of the articulation root's local +X axis in world
    space, from a world orientation quaternion - the standard extraction formula, not just
    atan2(qz, qw), so small roll/pitch (e.g. from suspension-ish wobble while driving) doesn't
    throw it off. This is NOT the direction the robot actually faces/drives - see
    ROBOT_FORWARD_OFFSET_RAD below.
    """
    w, x, y, z = orientation_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# The Galbot G1 asset's root local +X axis (what robot_heading_yaw measures) is not the
# direction it actually drives forward in - confirmed empirically by driving the robot
# straight (command=[1,0,0]) and comparing its actual displacement angle to robot_heading_yaw:
# they differ by a constant -pi/2 (heading was consistently 90deg ahead of the real movement
# direction). Subtract this offset to get the angle the robot actually faces/drives toward, for
# the map view (see build_scene/main below and static/viewer.js's map triangle).
ROBOT_FORWARD_OFFSET_RAD = np.pi / 2.0


def build_scene(stage, assets_root_path: str) -> list:
    """Builds the table + landmark cubes and returns their footprints (world xy, meters) for
    the browser's 2D map view (see FrameStore.update_world_state / static/viewer.js) - kept in
    sync with what's actually placed here rather than duplicating positions in two places.
    """
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1000.0)
    distant_light = UsdLux.DistantLight.Define(stage, "/World/SunLight")
    distant_light.CreateIntensityAttr(3000.0)

    add_reference_to_stage(usd_path=assets_root_path + TABLE_ASSET, prim_path="/World/Table")
    SingleXFormPrim("/World/Table", position=np.array([3.0, 1.5, 0.0]))

    bbox_cache = bounds_utils.create_bbox_cache()
    table_aabb = compute_world_aabb(bbox_cache, "/World/Table")
    objects = [
        {
            "label": "table",
            "x": float((table_aabb[0] + table_aabb[3]) / 2.0),
            "y": float((table_aabb[1] + table_aabb[4]) / 2.0),
            "hw": float((table_aabb[3] - table_aabb[0]) / 2.0),
            "hh": float((table_aabb[4] - table_aabb[1]) / 2.0),
            "color": "#8a8a8a",
        }
    ]

    # A few simple landmarks so the camera feed and point cloud have something to show besides
    # an empty floor.
    for i, (x, y, color, label) in enumerate(
        [
            (2.0, -1.5, (0.8, 0.2, 0.2), "box0"),
            (3.5, -0.5, (0.2, 0.8, 0.2), "box1"),
            (1.5, 2.5, (0.2, 0.2, 0.8), "box2"),
        ]
    ):
        cube = UsdGeom.Cube.Define(stage, f"/World/Landmark{i}")
        cube.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.3))
        cube.AddScaleOp().Set(Gf.Vec3f(0.3, 0.3, 0.3))
        cube.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2.0)
        objects.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "hw": 0.3,
                "hh": 0.3,
                "color": "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color)),
            }
        )

    return objects


def main() -> None:
    os.makedirs("captures", exist_ok=True)

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not resolve the Isaac Sim assets root path (check network access).")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = get_current_stage()

    scene_objects = build_scene(stage, assets_root_path)
    build_room(stage)
    add_reference_to_stage(usd_path=assets_root_path + ROBOT_ASSET, prim_path=ROBOT_PRIM)

    camera = Camera(
        prim_path=f"{ROBOT_PRIM}/OmniChassis/base_link/front_camera",
        frequency=20,
        resolution=(640, 480),
    )
    # RPLIDAR_S2E is a real single-plane 2D lidar (that's the actual hardware's design, not a
    # sim limitation) - it only ever scans one horizontal ring, hence the "line" of points, not
    # a proper 3D cloud. Ouster OS1 (32-channel) is a real multi-line 3D lidar - 32 stacked
    # scan rings - so the point cloud gets vertical structure (floor, walls, table legs, etc.)
    # instead of one flat slice.
    lidar = LidarRtx(
        prim_path=f"{LIDAR_MOUNT_PATH}/RtxLidar",
        name="chassis_lidar",
        config_file_name="OS1_REV6_32ch10hz512res",
    )

    robot = SingleArticulation(ROBOT_PRIM)

    world.reset()
    robot.initialize()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    lidar.initialize()
    lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

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

    left_arm_swing_rate = arm_swing_rate("left", args.arm_speed)
    right_arm_swing_rate = arm_swing_rate("right", args.arm_speed)
    left_arm_swing_fraction = 0.0
    right_arm_swing_fraction = 0.0
    torso_height_fraction = 0.0  # 0 = TORSO_UP_POSE, 1 = TORSO_DOWN_POSE
    hand_updown_rad = 0.0  # raw joint4 angle, shared by both arms
    gripper_rad = 0.0  # raw gripper drive-joint angle, shared by both hands

    frame_store = FrameStore()
    run_in_background(frame_store, host=args.host, port=args.port)

    held_keys: set = set()
    reset_requested = False

    def on_keyboard_event(event, *_args, **_kwargs) -> bool:
        nonlocal reset_requested
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                reset_requested = True
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
    print("  Release any key to freeze the arms in place. R resets. Close the window to exit.")

    # One-time diagnostic (see below) - list so the closure can mutate it without `nonlocal`.
    diagnostics_printed = [False]
    physics_dt = world.get_physics_dt()

    while simulation_app.is_running():
        if reset_requested:
            reset_requested = False
            world.reset()
            robot.initialize()
            left_arm_swing_fraction = 0.0
            right_arm_swing_fraction = 0.0
            torso_height_fraction = 0.0
            hand_updown_rad = 0.0
            gripper_rad = 0.0
            continue

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

        rgba = camera.get_rgba()
        if rgba is not None:
            frame_store.update_rgb(rgba)
        depth = camera.get_depth()
        if depth is not None:
            frame_store.update_depth(depth)
        point_cloud = lidar.get_current_frame().get("IsaacExtractRTXSensorPointCloudNoAccumulator", {})
        points = point_cloud.get("data") if point_cloud else None
        if points is not None and len(points):
            frame_store.update_points(points)
            if not diagnostics_printed[0]:
                diagnostics_printed[0] = True
                pts = np.asarray(points)
                print(
                    f"[lidar diagnostics, one-time] shape={pts.shape} dtype={pts.dtype} "
                    f"min={pts.min(axis=0) if pts.ndim == 2 else pts.min()} "
                    f"max={pts.max(axis=0) if pts.ndim == 2 else pts.max()} "
                    f"first_3_points={pts[:3].tolist() if pts.ndim == 2 else pts[:9].tolist()}"
                )

        robot_pos, robot_rot = robot.get_world_pose()
        frame_store.update_world_state(
            {
                "room": {"min": list(ROOM_MIN), "max": list(ROOM_MAX)},
                "objects": scene_objects,
                "robot": {
                    "x": float(robot_pos[0]),
                    "y": float(robot_pos[1]),
                    "heading": robot_heading_yaw(robot_rot) - ROBOT_FORWARD_OFFSET_RAD,
                },
            }
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
