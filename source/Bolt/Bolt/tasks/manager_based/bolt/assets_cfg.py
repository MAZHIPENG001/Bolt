"""Asset configurations for the Inreal V2 soccer task."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[6]
INREAL_V2_USD_PATH = (
    _PROJECT_ROOT / "data/assets/Inreal_v2/usd/inreal_v2_entity2_0528_robot_contact.usda"
)

_NATURAL_FREQ = 7.0 * 2.0 * math.pi
_DAMPING_RATIO = 2.0


def _pd_gains(armature: float) -> tuple[float, float]:
    """BeyondAmp_Mjlab's inertia-scaled 7 Hz, damping-ratio-2 gains."""
    stiffness = armature * _NATURAL_FREQ**2
    damping = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
    return stiffness, damping


_L1_GAINS = _pd_gains(0.291)
_L2_GAINS = _pd_gains(0.181)
_L3_GAINS = _pd_gains(0.1516)
_A2_GAINS = _pd_gains(0.036)


INREAL_V2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path="/pelvis",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(INREAL_V2_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            # Keep overlap correction from acting like an artificial kick.
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        # Override the larger importer/default contact envelope on all robot
        # colliders.  Together with the ball offset this limits visible gaps to
        # roughly 4 mm instead of centimetres.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
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
            "waist_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "hip_pitch": DCMotorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            stiffness=_L1_GAINS[0],
            damping=_L1_GAINS[1],
            effort_limit=586.85,
            effort_limit_sim=586.85,
            velocity_limit=13.12,
            saturation_effort=586.85,
            armature=0.291,
        ),
        "hip_roll": DCMotorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            stiffness=_L2_GAINS[0],
            damping=_L2_GAINS[1],
            effort_limit=336.0,
            effort_limit_sim=336.0,
            velocity_limit=12.9,
            saturation_effort=336.0,
            armature=0.181,
        ),
        "hip_yaw": DCMotorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            stiffness=_L2_GAINS[0],
            damping=_L2_GAINS[1],
            effort_limit=336.0,
            effort_limit_sim=336.0,
            velocity_limit=12.9,
            saturation_effort=336.0,
            armature=0.181,
        ),
        "knee": DCMotorCfg(
            joint_names_expr=[".*_knee_joint"],
            stiffness=_L3_GAINS[0],
            damping=_L3_GAINS[1],
            effort_limit=338.4,
            effort_limit_sim=338.4,
            velocity_limit=24.3,
            saturation_effort=338.4,
            armature=0.1516,
        ),
        "ankle_pitch": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            stiffness=_L3_GAINS[0],
            damping=_L3_GAINS[1],
            effort_limit=338.4,
            effort_limit_sim=338.4,
            velocity_limit=24.3,
            saturation_effort=338.4,
            armature=0.1516,
        ),
        "ankle_roll": DCMotorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=1.8,
            effort_limit=38.4,
            effort_limit_sim=38.4,
            velocity_limit=40.75,
            saturation_effort=38.4,
            armature=0.001584,
        ),
        "waist": DCMotorCfg(
            joint_names_expr=["waist_.*_joint"],
            stiffness=_L2_GAINS[0],
            damping=_L2_GAINS[1],
            effort_limit=336.0,
            effort_limit_sim=336.0,
            velocity_limit=12.875,
            saturation_effort=336.0,
            armature=0.181,
        ),
        "shoulder_pitch": DCMotorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint"],
            stiffness=_L2_GAINS[0],
            damping=_L2_GAINS[1],
            effort_limit=336.0,
            effort_limit_sim=336.0,
            velocity_limit=12.9,
            saturation_effort=336.0,
            armature=0.181,
        ),
        "shoulder_roll": DCMotorCfg(
            joint_names_expr=[".*_shoulder_roll_joint"],
            stiffness=_L2_GAINS[0],
            damping=_L2_GAINS[1],
            effort_limit=336.0,
            effort_limit_sim=336.0,
            velocity_limit=12.9,
            saturation_effort=336.0,
            armature=0.181,
        ),
        "elbow": DCMotorCfg(
            joint_names_expr=[".*_elbow_joint"],
            stiffness=_A2_GAINS[0],
            damping=_A2_GAINS[1],
            effort_limit=112.2,
            effort_limit_sim=112.2,
            velocity_limit=33.765,
            saturation_effort=112.2,
            armature=0.036,
        ),
    },
)
"""Inreal V2 articulation with the PD and motor limits used by the soccer controller."""


SOCCER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soccer",
    spawn=sim_utils.SphereCfg(
        radius=0.11,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=30.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.6,
            restitution=0.54,
            friction_combine_mode="average",
            restitution_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.9, 0.08, 0.02),
            roughness=0.45,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 1.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)
"""Soccer ball approximation matching the 0.43 kg, 0.11 m MuJoCo ball."""
