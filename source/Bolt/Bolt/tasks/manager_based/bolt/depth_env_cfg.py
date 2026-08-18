"""Depth-camera variant of the Bolt soccer juggling environment."""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .bolt_env_cfg import (
    HISTORY_LENGTH,
    BoltEnvCfg,
    BoltSceneCfg,
    ObservationsCfg,
    RewardsCfg,
)


@configclass
class BoltDepthSceneCfg(BoltSceneCfg):
    """Base scene plus a batched depth camera rigidly attached to the torso."""

    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/torso_link/DepthCamera",
        # The robot has no head link.  This pose approximates a chest/head
        # camera and looks 45 degrees downward in the robot's sagittal plane.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0.0, 0.12),
            rot=(math.cos(math.pi / 8.0), 0.0, math.sin(math.pi / 8.0), 0.0),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.12, 4.0),
        ),
        width=96,
        height=72,
        update_latest_camera_pose=True,
        depth_clipping_behavior="none",
    )


@configclass
class DepthObservationsCfg(ObservationsCfg):
    """Depth-derived actor state and privileged ground-truth critic state."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        soccer_pos_b = ObsTerm(
            func=mdp.depth_soccer_position_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "camera_cfg": SceneEntityCfg("depth_camera"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )
        soccer_vel_b = ObsTerm(
            func=mdp.depth_soccer_velocity_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "camera_cfg": SceneEntityCfg("depth_camera"),
            },
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=HISTORY_LENGTH,
        )
        next_kick_foot = ObsTerm(func=mdp.next_kick_foot_command_obs)
        feet_to_ball_b = ObsTerm(
            func=mdp.depth_feet_face_to_ball_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "camera_cfg": SceneEntityCfg("depth_camera"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
        )


@configclass
class DepthRewardsCfg(RewardsCfg):
    """Teacher reward formulas evaluated from depth-derived ball state."""


@configclass
class BoltDepthEnvCfg(BoltEnvCfg):
    """Deployable actor observations backed by a torso depth camera."""

    # Tiled rendering is substantially heavier than state-only simulation.
    scene: BoltDepthSceneCfg = BoltDepthSceneCfg(num_envs=256, env_spacing=5.0, replicate_physics=True)
    observations: DepthObservationsCfg = DepthObservationsCfg()
    rewards: DepthRewardsCfg = DepthRewardsCfg()
    # Reward helpers use this explicit contract instead of checking whether a
    # simulator ball happens to exist in the scene.
    depth_reward_inputs: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rerender_on_reset = True
