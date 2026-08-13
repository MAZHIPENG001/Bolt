"""IsaacLab port of the BeyondAmp_Mjlab Inreal V2 soccer task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .assets_cfg import INREAL_V2_CFG, SOCCER_CFG


HISTORY_LENGTH = 5
UPPER_BODY_NAMES = (
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
)
ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
)
UPPER_BODY_JOINTS = ("waist_pitch_joint", "waist_yaw_joint", *ARM_JOINTS)
LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
LATERAL_LEG_JOINTS = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)


@configclass
class BoltSceneCfg(InteractiveSceneCfg):
    """Ground, Inreal V2 robot, soccer ball, and contact sensors."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="max",
            ),
        ),
    )

    robot: ArticulationCfg = INREAL_V2_CFG
    soccer: RigidObjectCfg = SOCCER_CFG

    robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )
    left_foot_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/left_ankle_roll_link",
        update_period=0.0,
        history_length=4,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Soccer"],
    )
    right_foot_ball_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/right_ankle_roll_link",
        update_period=0.0,
        history_length=4,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Soccer"],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


@configclass
class ActionsCfg:
    """Reference soccer-controller position action scales."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_hip_pitch_joint": 0.25,
            ".*_hip_roll_joint": 0.25,
            ".*_hip_yaw_joint": 0.20,
            ".*_knee_joint": 0.25,
            ".*_ankle_pitch_joint": 0.12,
            ".*_ankle_roll_joint": 0.10,
            "waist_.*_joint": 0.10,
            ".*_shoulder_.*_joint": 0.10,
            ".*_elbow_joint": 0.10,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Five-frame actor history and asymmetric privileged critic state."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"body_name": "torso_link"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=HISTORY_LENGTH,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"body_name": "torso_link"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=HISTORY_LENGTH,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=HISTORY_LENGTH,
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=HISTORY_LENGTH)
        soccer_pos_b = ObsTerm(
            func=mdp.soccer_position_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        soccer_vel_b = ObsTerm(
            func=mdp.soccer_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=HISTORY_LENGTH,
        )
        feet_pos_b = ObsTerm(
            func=mdp.feet_face_pos_b,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        feet_projected_gravity = ObsTerm(
            func=mdp.feet_face_projected_gravity,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        feet_to_ball_b = ObsTerm(
            func=mdp.feet_face_to_ball_b,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        root_pos = ObsTerm(func=mdp.root_pos_w, history_length=HISTORY_LENGTH)
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"body_name": "torso_link"},
            history_length=HISTORY_LENGTH,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"body_name": "torso_link"},
            history_length=HISTORY_LENGTH,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=HISTORY_LENGTH)
        last_action = ObsTerm(func=mdp.last_action, history_length=HISTORY_LENGTH)
        base_lin_vel = ObsTerm(func=mdp.base_link_lin_vel_b)
        soccer_pos_b = ObsTerm(
            func=mdp.soccer_position_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            history_length=HISTORY_LENGTH,
        )
        soccer_vel_b = ObsTerm(
            func=mdp.soccer_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            history_length=HISTORY_LENGTH,
        )
        next_kick_foot = ObsTerm(func=mdp.next_kick_foot_obs)
        feet_pos_b = ObsTerm(func=mdp.feet_face_pos_b)
        feet_projected_gravity = ObsTerm(func=mdp.feet_face_projected_gravity)
        feet_lin_vel_b = ObsTerm(func=mdp.feet_face_lin_vel_b)
        feet_ang_vel_b = ObsTerm(func=mdp.feet_ang_vel_b)
        feet_to_ball_b = ObsTerm(func=mdp.feet_face_to_ball_b)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Reset randomization and sim-to-real domain randomization."""

    soccer_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.40, 0.65),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    foot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("left_ankle_roll_link", "right_ankle_roll_link")
            ),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    soccer_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("soccer"),
            "mass_distribution_params": (0.42, 0.45),
            "operation": "abs",
        },
    )
    soccer_inertia = EventTerm(
        func=mdp.set_soccer_diagonal_inertia,
        mode="startup",
        params={
            "soccer_cfg": SceneEntityCfg("soccer"),
            "reference_mass": 0.43,
            "reference_diagonal_inertia": 0.0035,
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    knee_ankle_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(".*_knee_joint", ".*_ankle_pitch_joint")
            ),
            "friction_distribution_params": (0.2, 1.2),
            "operation": "scale",
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "position_distribution_params": (-0.03, 0.03),
        },
    )
    foot_face_frames = EventTerm(
        func=mdp.randomize_virtual_foot_faces,
        mode="startup",
        params={
            "position_range": (-0.01, 0.01),
            "rotation_range": (-0.02, 0.02),
        },
    )

    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_around_nominal,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "joint_pos_dict": {
                ".*_hip_pitch_joint": -0.5,
                ".*_hip_yaw_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_knee_joint": 1.0,
                ".*_ankle_pitch_joint": -0.5,
                ".*_ankle_roll_joint": 0.0,
                ".*_elbow_joint": -1.5,
                "left_shoulder_roll_joint": 0.5,
                "left_shoulder_pitch_joint": 0.3,
                "right_shoulder_roll_joint": -0.5,
                "right_shoulder_pitch_joint": 0.3,
            },
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    reset_soccer = EventTerm(
        func=mdp.reset_soccer_drop,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "soccer_cfg": SceneEntityCfg("soccer"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    """Phase 1a weights; the curriculum switches these to Phase 1b in-place."""

    foot_ball_distance = RewTerm(
        func=mdp.foot_face_ball_distance_directional,
        weight=mdp.PHASE_1A_WEIGHTS["foot_ball_distance"],
        params={"std": 0.15, "unreachable_height": 0.3, "unreachable_vz": 0.2, "kick_zone_max_height": 0.4},
    )
    robot_facing_ball = RewTerm(
        func=mdp.robot_facing_ball_reward,
        weight=mdp.PHASE_1A_WEIGHTS["robot_facing_ball"],
        params={"min_ball_dist": 0.4},
    )
    next_kick_foot_height = RewTerm(
        func=mdp.next_kick_foot_height_reward,
        weight=mdp.PHASE_1A_WEIGHTS["next_kick_foot_height"],
        params={"target_height": 0.3, "max_track_height": 0.4, "std": 0.12},
    )
    kick = RewTerm(
        func=mdp.kick_reward,
        weight=mdp.PHASE_1A_WEIGHTS["kick"],
        params={"same_foot_scale": 0.0},
    )
    alternating_kick = RewTerm(
        func=mdp.alternating_kick_reward,
        weight=mdp.PHASE_1A_WEIGHTS["alternating_kick"],
    )
    kick_quality = RewTerm(
        func=mdp.kick_quality_reward,
        weight=mdp.PHASE_1A_WEIGHTS["kick_quality"],
        params={"target_vz": 3.2, "std": 1.0, "min_vz": 0.5},
    )
    juggle_streak = RewTerm(
        func=mdp.juggle_streak_reward,
        weight=mdp.PHASE_1A_WEIGHTS["juggle_streak"],
        params={"max_count": 5},
    )
    ball_height = RewTerm(
        func=mdp.ball_height_reward,
        weight=mdp.PHASE_1A_WEIGHTS["ball_height"],
        params={"target_height": 0.8, "std": 0.15},
    )
    ball_approach_vel = RewTerm(
        func=mdp.ball_approach_velocity_reward,
        weight=mdp.PHASE_1A_WEIGHTS["ball_approach_vel"],
        params={"activate_dist": 0.95, "deactivate_dist": 1.5, "max_approach_vel": 0.6},
    )
    ball_upward_vel = RewTerm(
        func=mdp.ball_upward_velocity_reward,
        weight=mdp.PHASE_1A_WEIGHTS["ball_upward_vel"],
        params={"target_vel": 3.2, "std": 1.0, "max_vel": 3.5, "min_vel": 0.5},
    )
    ball_horiz_vel_penalty = RewTerm(
        func=mdp.ball_horizontal_velocity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ball_horiz_vel_penalty"],
        params={"max_horiz_vel": 0.5},
    )
    kick_force_penalty = RewTerm(
        func=mdp.kick_force_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["kick_force_penalty"],
        params={"max_speed": 3.5},
    )
    robot_upright = RewTerm(
        func=mdp.robot_upright_reward,
        weight=mdp.PHASE_1A_WEIGHTS["robot_upright"],
        params={"juggle_discount": 0.9},
    )
    same_foot_kick_penalty = RewTerm(
        func=mdp.same_foot_kick_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["same_foot_kick_penalty"],
    )
    wrong_foot_proximity_penalty = RewTerm(
        func=mdp.wrong_foot_proximity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["wrong_foot_proximity_penalty"],
        params={"sigma": 0.16, "abs_threshold": 0.1, "activate_height": 0.5},
    )
    double_foot_proximity_penalty = RewTerm(
        func=mdp.double_foot_proximity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["double_foot_proximity_penalty"],
        params={"max_dist": 0.22},
    )
    both_feet_clamp_penalty = RewTerm(
        func=mdp.both_feet_clamp_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["both_feet_clamp_penalty"],
        params={"saturate_steps": 8},
    )
    ball_stationary_near_foot = RewTerm(
        func=mdp.ball_stationary_near_foot_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ball_stationary_near_foot"],
        params={"sigma": 0.4, "vel_threshold": 0.5},
    )
    ball_hold_duration = RewTerm(
        func=mdp.ball_hold_duration_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ball_hold_duration"],
        params={"max_steps": 12},
    )
    stable_standing = RewTerm(
        func=mdp.stable_standing_reward,
        weight=mdp.PHASE_1A_WEIGHTS["stable_standing"],
        params={"vel_std": 0.3, "ang_vel_std": 0.5, "base_fraction": 0.2},
    )
    torso_upright = RewTerm(
        func=mdp.torso_upright_reward,
        weight=mdp.PHASE_1A_WEIGHTS["torso_upright"],
        params={"torso_body_name": "torso_link", "base_fraction": 0.9},
    )
    torso_backward_lean_penalty = RewTerm(
        func=mdp.torso_backward_lean_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["torso_backward_lean_penalty"],
        params={"torso_body_name": "torso_link", "juggle_scale": 1.0, "standing_scale": 1.0},
    )
    juggling_yaw_penalty = RewTerm(
        func=mdp.juggling_yaw_velocity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["juggling_yaw_penalty"],
        params={"max_yaw_vel": 0.5},
    )
    double_contact_penalty = RewTerm(
        func=mdp.double_contact_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["double_contact_penalty"],
    )
    upper_body_joint_penalty = RewTerm(
        func=mdp.upper_body_joint_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["upper_body_joint_penalty"],
        params={"joint_names": UPPER_BODY_JOINTS, "std": 0.15, "juggle_discount": 0.7},
    )
    arm_action_rate_penalty = RewTerm(
        func=mdp.arm_action_rate_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["arm_action_rate_penalty"],
        params={"joint_names": ARM_JOINTS, "juggle_discount": 0.7},
    )
    arm_symmetry_penalty = RewTerm(
        func=mdp.arm_symmetry_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["arm_symmetry_penalty"],
        params={
            "left_joints": ARM_JOINTS[:3],
            "right_joints": ARM_JOINTS[3:],
            "sign_flip": (1.0, -1.0, 1.0),
            "juggle_discount": 0.7,
        },
    )
    ankle_pitch_vel_penalty = RewTerm(
        func=mdp.joint_group_velocity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ankle_pitch_vel_penalty"],
        params={
            "joint_names": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
            "max_vel": 6.0,
            "juggle_discount": 0.7,
        },
    )
    ankle_roll_vel_penalty = RewTerm(
        func=mdp.joint_group_velocity_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ankle_roll_vel_penalty"],
        params={
            "joint_names": ("left_ankle_roll_joint", "right_ankle_roll_joint"),
            "max_vel": 2.5,
            "juggle_discount": 1.0,
        },
    )
    juggling_lateral_leg_pose_penalty = RewTerm(
        func=mdp.juggling_lateral_leg_pose_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["juggling_lateral_leg_pose_penalty"],
        params={"joint_names": LATERAL_LEG_JOINTS, "std": 0.12, "juggle_discount": 1.0},
    )
    ball_drop_idle_penalty = RewTerm(
        func=mdp.ball_drop_idle_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ball_drop_idle_penalty"],
        params={"max_height": 0.5, "min_height": 0.15, "min_approach_vel": 0.1},
    )
    leg_action_rate_penalty = RewTerm(
        func=mdp.leg_action_rate_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["leg_action_rate_penalty"],
        params={"joint_names": LEG_JOINTS},
    )
    ball_alive_bonus = RewTerm(
        func=mdp.ball_alive_bonus,
        weight=mdp.PHASE_1A_WEIGHTS["ball_alive_bonus"],
    )
    ball_grounding_penalty = RewTerm(
        func=mdp.ball_grounding_penalty,
        weight=mdp.PHASE_1A_WEIGHTS["ball_grounding_penalty"],
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=mdp.PHASE_1A_WEIGHTS["action_rate_l2"],
    )
    action_rate_2nd_l2 = RewTerm(
        func=mdp.action_rate_2nd_l2,
        weight=mdp.PHASE_1A_WEIGHTS["action_rate_2nd_l2"],
    )
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=mdp.PHASE_1A_WEIGHTS["joint_limit"],
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=mdp.PHASE_1A_WEIGHTS["undesired_contacts"],
        params={
            "sensor_cfg": SceneEntityCfg("robot_contacts", body_names=UPPER_BODY_NAMES),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("robot_contacts", body_names=UPPER_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    ball_on_ground = DoneTerm(
        func=mdp.soccer_ball_on_ground_phase1b,
        params={"ball_radius": 0.11},
    )
    ball_too_far = DoneTerm(
        func=mdp.soccer_ball_too_far_phase1b,
        params={"max_distance": 4.0},
    )
    ball_too_high = DoneTerm(
        func=mdp.soccer_ball_too_high_phase1b,
        params={"max_height": 3.0},
    )
    ball_behind_robot = DoneTerm(
        func=mdp.soccer_ball_behind_robot_phase1b,
        params={"min_forward_dist": -0.75},
    )
    single_foot_bias = DoneTerm(
        func=mdp.single_foot_bias_terminate,
        params={"same_foot_threshold": 3},
    )
    both_feet_clamp = DoneTerm(
        func=mdp.both_feet_clamp_terminate,
        params={"min_steps": 20},
    )


@configclass
class CurriculumCfg:
    phase_1a_to_1b = CurrTerm(
        func=mdp.phase1a_to_1b_curriculum,
        params={
            "alternated_kick_ema_threshold": 0.001,
            "per_foot_kick_ema_threshold": 0.001,
            "foot_balance_threshold": 0.001,
            "episode_len_ema_threshold": 300.0,
            "min_global_steps": 240_000,
            "ema_alpha": 0.005,
            "initial_phase": 0,
        },
    )


@configclass
class BoltEnvCfg(ManagerBasedRLEnvCfg):
    """Reference-aligned privileged teacher task for soccer juggling."""

    # The environments contain the same assets, so physics replication avoids
    # independently parsing thousands of identical robot/ball scenes.
    scene: BoltSceneCfg = BoltSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 15.0
        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.9)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.05

        # PhysX GPU buffer
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
