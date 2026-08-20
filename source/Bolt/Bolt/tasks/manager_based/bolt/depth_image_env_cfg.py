"""End-to-end raw-depth-image variant of the Bolt soccer environment."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .bolt_env_cfg import HISTORY_LENGTH, BoltEnvCfg, ObservationsCfg
from .depth_env_cfg import BoltDepthSceneCfg


@configclass
class RawDepthImageObservationsCfg(ObservationsCfg):
    """Actor observations containing proprioception and raw depth pixels only."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Keep the same proprioceptive inputs and five-step history as the
        # state-based policy.  The two depth frames let the actor infer motion
        # from images without a hand-engineered ball-velocity estimator.
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
        raw_depth = ObsTerm(
            func=mdp.raw_depth_image,
            params={"camera_cfg": SceneEntityCfg("depth_camera")},
            history_length=2,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BoltDepthImageEnvCfg(BoltEnvCfg):
    """Soccer task with an actor trained end-to-end from raw depth pixels.

    Rewards and the training-only critic keep the teacher state.  In contrast
    to :class:`BoltDepthEnvCfg`, this task never invokes the depth ball
    detector: its actor must infer ball geometry and motion from raw images.
    """

    scene: BoltDepthSceneCfg = BoltDepthSceneCfg(num_envs=128, env_spacing=5.0, replicate_physics=True)
    observations: RawDepthImageObservationsCfg = RawDepthImageObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rerender_on_reset = True
