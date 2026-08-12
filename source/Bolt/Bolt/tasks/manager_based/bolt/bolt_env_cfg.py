"""Manager-based teacher environment for Inreal V2 soccer juggling."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .assets_cfg import INREAL_V2_CFG, SOCCER_CFG


@configclass
class BoltSceneCfg(InteractiveSceneCfg):
    """Ground, Inreal V2 robot, and one soccer ball per environment."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="max",
            ),
        ),
    )

    robot: ArticulationCfg = INREAL_V2_CFG
    soccer: RigidObjectCfg = SOCCER_CFG

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1000.0),
    )


@configclass
class ActionsCfg:
    """Joint target actions centered on the crouched initial pose."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*_hip_.*_joint": 0.25,
            ".*_knee_joint": 0.25,
            ".*_ankle_.*_joint": 0.25,
            "waist_.*_joint": 0.15,
            ".*_shoulder_.*_joint": 0.12,
            ".*_elbow_joint": 0.12,
        },
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Teacher observations and privileged critic state."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-20.0, 20.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.005, n_max=0.005),
            clip=(-3.0, 3.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-40.0, 40.0),
            scale=0.05,
        )
        last_action = ObsTerm(func=mdp.last_action)
        soccer_pos_b = ObsTerm(
            func=mdp.soccer_position_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            noise=Unoise(n_min=-0.005, n_max=0.005),
            clip=(-3.0, 3.0),
        )
        soccer_vel_b = ObsTerm(
            func=mdp.soccer_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-20.0, 20.0),
            scale=0.25,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-10.0, 10.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-20.0, 20.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-3.0, 3.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-40.0, 40.0), scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        soccer_pos_b = ObsTerm(
            func=mdp.soccer_position_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            clip=(-3.0, 3.0),
        )
        soccer_vel_b = ObsTerm(
            func=mdp.soccer_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
            clip=(-20.0, 20.0),
            scale=0.25,
        )
        soccer_height = ObsTerm(func=mdp.soccer_height, params={"soccer_cfg": SceneEntityCfg("soccer")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Reset randomization for the robot and ball."""

    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # Keep joint_ids as slice(None). Supplying an explicit list together
            # with a multi-env tensor triggers paired advanced indexing in the
            # IsaacLab 0.41 reset helper.
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.05, 0.05),
        },
    )

    reset_soccer = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("soccer"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.08, 0.08), "z": (-0.05, 0.05)},
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.1, 0.1),
                "roll": (-2.0, 2.0),
                "pitch": (-2.0, 2.0),
                "yaw": (-2.0, 2.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Initial dense rewards for learning the first juggling contact."""

    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    ball_height = RewTerm(
        func=mdp.ball_height_reward,
        weight=4.0,
        params={"minimum_height": 0.13, "target_height": 0.9, "soccer_cfg": SceneEntityCfg("soccer")},
    )
    ball_centered = RewTerm(
        func=mdp.ball_centered_xy_exp,
        weight=2.0,
        params={"std": 0.8, "robot_cfg": SceneEntityCfg("robot"), "soccer_cfg": SceneEntityCfg("soccer")},
    )
    foot_ball_proximity = RewTerm(
        func=mdp.feet_to_ball_proximity_exp,
        weight=2.5,
        params={
            "std": 0.35,
            "ball_radius": 0.11,
            "foot_face_offset": (0.17, 0.0, -0.005),
            "robot_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
            ),
            "soccer_cfg": SceneEntityCfg("soccer"),
        },
    )
    ball_upward_velocity = RewTerm(
        func=mdp.ball_upward_velocity,
        weight=1.5,
        params={"max_velocity": 4.0, "soccer_cfg": SceneEntityCfg("soccer")},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 1.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fallen = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.65, "asset_cfg": SceneEntityCfg("robot")},
    )
    robot_tilted = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.9, "asset_cfg": SceneEntityCfg("robot")},
    )
    ball_lost = DoneTerm(
        func=mdp.ball_lost,
        params={
            # The ball radius is 0.11 m, so its center rests at roughly z=0.11 m.
            # A threshold above that height causes a reset every time the ball
            # touches the ground, leaving an untrained policy only ~0.5 s to act.
            # Keep this below the resting center height and terminate only if the
            # ball falls through the floor (or travels too far horizontally).
            "minimum_height": 0.08,
            "maximum_horizontal_distance": 2.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "soccer_cfg": SceneEntityCfg("soccer"),
        },
    )


@configclass
class BoltEnvCfg(ManagerBasedRLEnvCfg):
    """Privileged teacher task used before replacing ball truth with vision."""

    # Conservative default for the local 8 GB GPU; override with --num_envs after profiling.
    scene: BoltSceneCfg = BoltSceneCfg(num_envs=256, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 12.0

        self.viewer.eye = (4.0, 4.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.9)

        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.05
