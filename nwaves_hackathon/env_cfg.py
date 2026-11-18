"""NWaves SO-101 hackathon environments (Isaac Lab 3.0, Isaac Sim 6.2): put the cube on the tray.

Scene: the SO-101 (Isaac Lab's system-identified model, with the official camera mount and its
calibrated WowRobo 2MP wrist camera) inside the workspace enclosure, NWaves.ai on the front bevel,
one 25 mm cube (a new colour every episode) and one black tray (a low open box), on a wooden table top. A scene
camera looks in from behind the arm's right
shoulder. Control runs at 30 Hz (physics 120 Hz) in all variants. Every episode starts at the SO-101
home pose (all arm joints 0).

Drawn again at every reset, from the ranges in ``cfg.randomization`` (see randomization.py): cube
position, yaw and colour, tray position and yaw, the look of the workspace (table top, wall and arm colours),
light intensity and colour temperature, the scene camera's
mounting pose, cube friction and mass, motor gains.

    NWaves-SO101-PickPlace-v0            the official task: joint-position actions, official randomisation,
                                         30 s, ends at success or timeout
    NWaves-SO101-PickPlace-Eval-v0       the scorer's copy of the official task (scripts/evaluate.py)
    NWaves-SO101-PickPlace-Kit-v0        fingertip-pose (IK) actions for the generator, the keyboard teleop
                                         and Mimic; randomisation from configs/datagen.yaml; 10 minute episodes
    NWaves-SO101-PickPlace-Kit-Mimic-v0  the kit environment with the Isaac Lab Mimic interface
"""
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg, UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

# Isaac Lab ships its SO-101 setup (the robot, its seat, the pose IK) inside one of its example packages; only the
# robot, the IK and the simulation settings are taken from there, the scene and the task below are ours.
from isaaclab_tasks.contrib.stack.config.so101.stack_joint_pos_env_cfg import SO101CubeStackEnvCfg as _IsaacLabSO101Cfg

from . import layout as L
from . import mdp as H
from . import randomization as R
from .surface import surface_usd_path
from .tray import tray_usd_path

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _look_at_quat_world(eye, target):
    """xyzw quaternion turning +X (Isaac Lab "world" camera convention) towards target, +Z up."""
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))          # positive = look down
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    # q = q_yaw(z) * q_pitch(y)
    return (-sy * sp, cy * sp, sy * cp, cy * cp)


def _box(name, xy, size, z_bottom, color, collide=True):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True) if collide else None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], z_bottom + size[2] / 2)),
    )


def _cube(name, xy):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], L.CUBE_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(L.CUBE_SIZE,) * 3,
            rigid_props=[
                UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
                PhysxRigidBodyCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                  max_depenetration_velocity=5.0, disable_gravity=False),
            ],
            mass_props=sim_utils.MassCfg(mass=L.CUBE_MASS),
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=0.9),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=L.COLORS["red"], roughness=0.6),   # set anew at every reset
        ),
    )


def _surface():
    """The table top: a picture laid on the enclosure's floor, see surface.py. It is a fixed (kinematic) body with
    no collision shape, so a reset can lay it down or keep it out of sight below the floor."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Surface",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, L.SURFACE_SHEET_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=surface_usd_path(),
            rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
        ),
    )


def _mat():
    """The dark mat of the "classic" look: a thin fixed (kinematic) plate on the enclosure's floor. A reset lays it
    down or, for an episode without the mat, keeps it out of sight below the floor (see mdp.reset_scene)."""
    size = (L.MAT_SIZE[0], L.MAT_SIZE[1], L.MAT_THICK)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(L.MAT_CENTER[0], L.MAT_CENTER[1], L.SURFACE_Z + L.MAT_THICK / 2),
                                                  rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
            collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=L.MAT_COLOR, roughness=0.7),
        ),
    )


def _tray(xy):
    """The tray: one fixed (kinematic) body, see tray.py. Resets move it."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(xy[0], xy[1], L.TABLE_Z), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=tray_usd_path(),
            rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True)],
        ),
    )


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot = None                    # Isaac Lab's own SO-101 setup (seated, with the wrist camera), set in the env config
    ee_frame = None

    enclosure = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Enclosure",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(ASSETS, "nwaves_box.usda")),
        # CAD work surface at z = 0.49846 m; placed so the arm sits at its mount (see layout.py)
        init_state=AssetBaseCfg.InitialStateCfg(pos=L.ENCLOSURE_POS, rot=L.ENCLOSURE_ROT_XYZW),
    )
    surface = _surface()
    mat = _mat()
    cube = _cube("PickCube", L.FIXED_RANDOMIZATION["cube_x"][:1] + L.FIXED_RANDOMIZATION["cube_y"][:1])
    tray = _tray(L.FIXED_RANDOMIZATION["tray_x"][:1] + L.FIXED_RANDOMIZATION["tray_y"][:1])

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=L.DOME_INTENSITY, color=(0.94, 0.96, 1.0)),
    )
    key_light = AssetBaseCfg(                       # soft key light, 32.5 deg from vertical toward the opening
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=L.KEY_INTENSITY, angle=4.0, color=(1.0, 0.94, 0.82)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.0, 0.27982901, 0.0, 0.96005040)),
    )
    roof_strip_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoofStrip_1",          # inside every enclosure, not shared
        spawn=sim_utils.CylinderLightCfg(length=L.ROOF_STRIP_LEN, radius=L.ROOF_STRIP_W / 2, intensity=L.STRIP_INTENSITY, normalize=True,
                                         color=L.STRIP_COLOR, enable_color_temperature=True, color_temperature=L.STRIP_TEMPERATURE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(L.ROOF_STRIPS[0][0], L.ROOF_STRIPS[0][1], L.ROOF_STRIP_Z)),
    )
    roof_strip_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoofStrip_2",
        spawn=sim_utils.CylinderLightCfg(length=L.ROOF_STRIP_LEN, radius=L.ROOF_STRIP_W / 2, intensity=L.STRIP_INTENSITY, normalize=True,
                                         color=L.STRIP_COLOR, enable_color_temperature=True, color_temperature=L.STRIP_TEMPERATURE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(L.ROOF_STRIPS[1][0], L.ROOF_STRIPS[1][1], L.ROOF_STRIP_Z)),
    )
    roof_strip_visual_1 = _box("RoofStripVisual_1", L.ROOF_STRIPS[0], (L.ROOF_STRIP_LEN, L.ROOF_STRIP_W, 0.008),
                               L.ROOF_STRIP_Z + 0.003, (1.0, 0.64, 0.32), collide=False)
    roof_strip_visual_2 = _box("RoofStripVisual_2", L.ROOF_STRIPS[1], (L.ROOF_STRIP_LEN, L.ROOF_STRIP_W, 0.008),
                               L.ROOF_STRIP_Z + 0.003, (1.0, 0.64, 0.32), collide=False)

    scene_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCam",
        update_period=0.0,
        width=L.IMAGE_W,
        height=L.IMAGE_H,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=L.SCENE_FOCAL_MM, horizontal_aperture=L.SCENE_H_APERTURE_MM,
                                         vertical_aperture=L.SCENE_V_APERTURE_MM, clipping_range=(0.01, 5.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=L.SCENE_EYE, rot=_look_at_quat_world(L.SCENE_EYE, L.SCENE_TARGET),
                                        convention="world"),
    )
    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",   # authored by the SO-101 sensors variant
        update_period=0.0,
        width=L.IMAGE_W,
        height=L.IMAGE_H,
        data_types=["rgb"],
        spawn=None,
    )

    view_cam = TiledCameraCfg(                      # third-person view for videos and renders; 10 Hz, never observed by policies
        prim_path="{ENV_REGEX_NS}/ViewCam",
        update_period=0.1,
        width=L.VIEW_W,
        height=L.VIEW_H,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=TiledCameraCfg.OffsetCfg(pos=L.VIEW_EYE, rot=_look_at_quat_world(L.VIEW_EYE, L.VIEW_TARGET),
                                        convention="world"),
    )

    # The table under the enclosure (visual only; the enclosure floor is the work surface)
    table_top = _box("TableTop", L.TABLE_CENTER, (L.TABLE_SIZE[0], L.TABLE_SIZE[1], 0.05), L.TABLE_TOP_Z - 0.05,
                     (0.56, 0.43, 0.31), collide=False)
    table_leg_1 = _box("TableLeg_1", L.TABLE_LEGS[0], (0.065, 0.065, L.TABLE_TOP_Z - 0.05 - L.FLOOR_Z), L.FLOOR_Z, (0.18, 0.18, 0.19), collide=False)
    table_leg_2 = _box("TableLeg_2", L.TABLE_LEGS[1], (0.065, 0.065, L.TABLE_TOP_Z - 0.05 - L.FLOOR_Z), L.FLOOR_Z, (0.18, 0.18, 0.19), collide=False)
    table_leg_3 = _box("TableLeg_3", L.TABLE_LEGS[2], (0.065, 0.065, L.TABLE_TOP_Z - 0.05 - L.FLOOR_Z), L.FLOOR_Z, (0.18, 0.18, 0.19), collide=False)
    table_leg_4 = _box("TableLeg_4", L.TABLE_LEGS[3], (0.065, 0.065, L.TABLE_TOP_Z - 0.05 - L.FLOOR_Z), L.FLOOR_Z, (0.18, 0.18, 0.19), collide=False)
    # The fixed scene camera's housing at its mount, behind the optical centre (see layout.SCENE_CAM_BODY_POS)
    scene_cam_body = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SceneCamBody",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.04, 0.04),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.035, 0.038, 0.045), roughness=0.45)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=L.SCENE_CAM_BODY_POS, rot=_look_at_quat_world(L.SCENE_EYE, L.SCENE_TARGET)),
    )

    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/PickCube"],
    )

@configclass
class PickPlaceObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=base_mdp.joint_pos)          # rad, robot joint order

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class VisualCfg(ObsGroup):
        scene = ObsTerm(func=base_mdp.image_rgb,
                           params={"sensor_cfg": SceneEntityCfg("scene_cam"), "data_type": "rgb", "normalize": False})
        wrist = ObsTerm(func=base_mdp.image_rgb,
                        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    visual: VisualCfg = VisualCfg()


@configclass
class LimitedJointPositionActionCfg(base_mdp.JointPositionActionCfg):
    class_type: type | str = "nwaves_hackathon.ik_action:LimitedJointPositionAction"


@configclass
class PickPlaceActionsCfg:
    """Six joint position targets [rad] in the order of layout.JOINTS. The arm targets pass the speed limiter of
    the real arm's runner (layout.MAX_STEP_DEG, layout.MAX_LEAD_DEG)."""
    joints = LimitedJointPositionActionCfg(asset_name="robot", joint_names=list(L.JOINTS), scale=1.0,
                                           use_default_offset=False, preserve_order=True)


@configclass
class PickPlaceEventCfg:
    """What happens at every reset. ``layout`` must stay first: it seeds the random numbers of the reset from
    (seed, episode number). The ranges are filled in from ``cfg.randomization`` by randomization.apply()."""
    layout = EventTerm(func=H.reset_scene, mode="reset", params={"ranges": None})
    reset_robot = EventTerm(func=base_mdp.reset_joints_by_offset, mode="reset",       # exactly the home pose
                            params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                    "asset_cfg": SceneEntityCfg("robot")})
    cube_friction = EventTerm(func=base_mdp.randomize_rigid_body_material, mode="reset",
                              params={"asset_cfg": SceneEntityCfg("cube"), "static_friction_range": (1.0, 1.0),
                                      "dynamic_friction_range": (0.85, 0.85), "restitution_range": (0.0, 0.05),
                                      "num_buckets": 32, "make_consistent": True})
    cube_mass = EventTerm(func=base_mdp.randomize_rigid_body_mass, mode="reset",
                          params={"asset_cfg": SceneEntityCfg("cube"), "mass_distribution_params": (1.0, 1.0),
                                  "operation": "scale"})
    motor_gains = EventTerm(func=base_mdp.randomize_actuator_gains, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot"), "stiffness_distribution_params": (1.0, 1.0),
                                    "damping_distribution_params": (1.0, 1.0), "operation": "scale"})


@configclass
class OfficialTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=H.episode_success, time_out=False)


@configclass
class KitTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=H.track_success, time_out=False)       # keeps the success state, never ends the episode


@configclass
class PickPlaceEnvCfg(_IsaacLabSO101Cfg):
    """The official task: joint-position actions (rad, the order of layout.JOINTS), official randomisation,
    30 s per episode, ends at success or timeout."""

    randomization: dict = None        # ranges of the reset, see randomization.py; None = the official ones

    def __post_init__(self):
        super().__post_init__()
        robot, ee_frame = self.scene.robot, self.scene.ee_frame
        self.scene = PickPlaceSceneCfg(num_envs=self.scene.num_envs, env_spacing=4.0, replicate_physics=False)
        self.scene.robot = robot
        self.scene.ee_frame = ee_frame
        # the robot model: the copy the lab holds, so that no server is needed; without one, Isaac Lab fetches it
        local_robot = L.local_robot_usd()
        if local_robot:
            self.scene.robot.spawn.usd_path = local_robot
        # every episode starts at the original SO-101 home (all arm joints 0), where the real arm parks
        self.scene.robot.init_state.joint_pos = {**dict(self.scene.robot.init_state.joint_pos), **L.HOME_RAD,
                                                 "gripper": L.HOME_GRIPPER_RAD}
        # USD-authored (system-identified) gains for the arm; the gripper keeps them too but gets the real
        # servo's torque cap, so a jaw closed on a cube squeezes like an STS3215 instead of exploding it.
        from isaaclab.actuators import ImplicitActuatorCfg
        self.scene.robot.actuators = {
            "arm": ImplicitActuatorCfg(joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                         "wrist_roll"], stiffness=None, damping=None),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper"], stiffness=None, damping=None,
                                           joint_effort_limit=L.GRIPPER_EFFORT_NM, joint_velocity_limit=4.0),
        }
        self.observations = PickPlaceObservationsCfg()
        self.actions = PickPlaceActionsCfg()
        self.events = PickPlaceEventCfg()
        R.apply(self, self.randomization or L.OFFICIAL_RANDOMIZATION)
        self.terminations = OfficialTerminationsCfg()
        self.decimation = 4                          # 30 Hz control
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = L.EPISODE_SECONDS
        self.num_rerenders_on_reset = 2              # the first pictures of an episode show the new scene, not the old one
        # the Isaac Sim window shows the first scene from the third-person view, however many scenes run side by side
        self.viewer.eye = L.VIEW_EYE
        self.viewer.lookat = L.VIEW_TARGET
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0


@configclass
class PickPlaceEvalEnvCfg(PickPlaceEnvCfg):
    """The scorer's copy of the official task. Identical on purpose; scripts/evaluate.py uses this id."""


from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg as _PoseIKCfg  # noqa: E402


@configclass
class SO101PoseIKIntegralActionCfg(_PoseIKCfg):
    class_type: type | str = "nwaves_hackathon.ik_action:SO101PoseIKIntegralAction"
    integral_gain: float = 0.03        # per physics step
    integral_limit: float = 0.12       # rad, per joint


def ik_pose_actions(integral: bool = True):
    """Absolute fingertip pose for the arm (Isaac Lab's SO-101 pose IK, plus an integral term so the fingertip really
    reaches the command, see ik_action.py) + joint position for the gripper.
    Action = [x, y, z, qx, qy, qz, qw, gripper_rad] in the robot root frame."""
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg
    from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import SO101PoseIKControllerCfg
    from isaaclab_tasks.contrib.stack.config.so101.stack_ik_abs_env_cfg import SO101IkActionsCfg

    arm_cls = SO101PoseIKIntegralActionCfg if integral else SO101PoseIKActionCfg
    return SO101IkActionsCfg(
        arm_action=arm_cls(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            body_offset=SO101PoseIKActionCfg.OffsetCfg(pos=L.TCP_OFFSET),     # command the fingertip point
            scale=1.0,
            controller=SO101PoseIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="adaptive_dls",
                ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                # a straight-down grasp with a chosen jaw yaw is exactly reachable by the 5 joints, so all of
                # them serve orientation (no wrist-only mask) and position dominates the trade-off
                orientation_weight=0.2,
                orientation_joint_names=None,
                joint_limit_avoidance_gain=0.3,
                joint_limit_avoidance_margin=0.2,
            ),
        ),
        gripper_action=base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=["gripper"], scale=1.0,
                                                      use_default_offset=False),
    )


# ----------------------------------------------------------------------------------------------------- the kit
@configclass
class PickPlaceKitEnvCfg(PickPlaceEnvCfg):
    """The data-collection environment: the same scene, the arm is driven by a fingertip pose (IK), the episode
    does not end at success, and the randomisation is yours: it is read from configs/datagen.yaml (or the file
    named by NWAVES_DATAGEN_CONFIG). Used by the generator, the keyboard teleop and Mimic."""

    def __post_init__(self):
        if self.randomization is None:
            self.randomization = R.load_config()["randomization"]
        super().__post_init__()
        self.actions = ik_pose_actions()
        self.terminations = KitTerminationsCfg()
        self.episode_length_s = 600.0


# ------------------------------------------------------------------------------------- Isaac Lab Mimic config
@configclass
class MimicTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=H.success_settling, time_out=False)


def _subtask(object_ref, signal, offset=(5, 10), strategy="nearest_neighbor_object", interp=20):
    from isaaclab.envs.mimic_env_cfg import SubTaskConfig
    return SubTaskConfig(object_ref=object_ref, subtask_term_signal=signal,
                         subtask_term_offset_range=offset if signal else (0, 0),
                         selection_strategy=strategy if object_ref else "random",
                         selection_strategy_kwargs={"nn_k": 3} if (object_ref and strategy == "nearest_neighbor_object") else {},
                         action_noise=0.003, num_interpolation_steps=interp, num_fixed_steps=0,
                         apply_noise_during_interpolation=False)


from isaaclab.envs.mimic_env_cfg import MimicEnvCfg  # noqa: E402


@configclass
class PickPlaceKitMimicEnvCfg(PickPlaceKitEnvCfg, MimicEnvCfg):
    """Pick and place for Isaac Lab Mimic: grasp the cube (poses relative to the cube), then carry it to the
    tray, set it down and let go (poses relative to the tray)."""

    def __post_init__(self):
        super().__post_init__()
        # Mimic records joint states and pose actions only (images come later from nwaves-mimic export): no cameras,
        # which makes annotation and generation several times faster and avoids rendering for nothing
        self.scene.scene_cam = None
        self.scene.wrist_cam = None
        self.scene.view_cam = None
        self.observations.visual = None
        self.terminations = MimicTerminationsCfg()
        dg = self.datagen_config
        dg.name = "nwaves_so101_pickplace"
        dg.generation_guarantee = True
        dg.generation_keep_failed = False
        dg.generation_num_trials = 100
        dg.generation_select_src_per_subtask = True
        dg.generation_transform_first_robot_pose = False
        dg.generation_interpolate_from_last_target_pose = True
        dg.seed = 1
        self.subtask_configs["so101"] = [_subtask("cube", "grasp"), _subtask("tray", None)]
