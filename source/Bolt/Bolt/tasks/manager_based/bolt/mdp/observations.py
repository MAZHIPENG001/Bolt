"""Teacher observations aligned with the BeyondAmp_Mjlab soccer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

from .juggle_state import LEFT_FOOT_BODY, RIGHT_FOOT_BODY, _body_id, get_juggle_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


FOOT_FACE_OFFSET = (0.17, 0.0, -0.005)
IMU_BODY = "torso_link"


def imu_ang_vel(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = IMU_BODY,
) -> torch.Tensor:
    """Torso angular velocity expressed in the reconstructed IMU frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = _body_id(robot, body_name)
    return quat_apply_inverse(
        robot.data.body_link_quat_w[:, body_id], robot.data.body_link_ang_vel_w[:, body_id]
    )


def imu_projected_gravity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = IMU_BODY,
) -> torch.Tensor:
    """World gravity direction expressed in the torso-mounted IMU frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_id = _body_id(robot, body_name)
    quat_w = robot.data.body_link_quat_w[:, body_id]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=quat_w.dtype)
    return quat_apply_inverse(quat_w, gravity.expand(env.num_envs, -1))


def base_link_lin_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root-link linear velocity expressed in the root-link frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_apply_inverse(robot.data.root_link_quat_w, robot.data.root_link_lin_vel_w)


def _foot_face_kinematics(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return foot-face positions, orientations, and linear velocities in world frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = [_body_id(robot, LEFT_FOOT_BODY), _body_id(robot, RIGHT_FOOT_BODY)]
    body_pos = robot.data.body_pos_w[:, ids]
    body_quat = robot.data.body_quat_w[:, ids]
    body_lin_vel = robot.data.body_link_lin_vel_w[:, ids]
    body_ang_vel = robot.data.body_link_ang_vel_w[:, ids]
    offset = getattr(env, "_bolt_foot_face_pos_offset", None)
    if offset is None:
        offset = torch.tensor(FOOT_FACE_OFFSET, device=env.device, dtype=body_pos.dtype)
        offset = offset.expand_as(body_pos)
    else:
        offset = offset.to(dtype=body_pos.dtype)
    offset_w = quat_apply(body_quat, offset)
    face_pos = body_pos + offset_w
    face_lin_vel = body_lin_vel + torch.linalg.cross(body_ang_vel, offset_w, dim=-1)
    quat_offset = getattr(env, "_bolt_foot_face_quat_offset", None)
    face_quat = body_quat if quat_offset is None else quat_mul(body_quat, quat_offset)
    return face_pos, face_quat, face_lin_vel


def _ball_observation_gate(
    env: ManagerBasedRLEnv,
    relative_position_b: torch.Tensor,
    ball_height: torch.Tensor,
    clamp_dist: float,
    ground_height: float,
    transition_width: float,
) -> torch.Tensor:
    """Reference observation-only smooth gate (separate from the reward gate)."""
    horizontal_distance = torch.linalg.vector_norm(relative_position_b[:, :2], dim=-1)
    distance_gate = 1.0 - torch.clamp(
        (horizontal_distance - (clamp_dist - transition_width)) / (transition_width + 1.0e-6),
        0.0,
        1.0,
    )
    ground_gate = torch.clamp(
        (ball_height - (ground_height - transition_width)) / (transition_width + 1.0e-6),
        0.0,
        1.0,
    )
    return distance_gate * ground_gate


def soccer_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    clamp_dist: float = 3.0,
    ground_height: float = 0.15,
    transition_width: float = 0.3,
) -> torch.Tensor:
    """Ball position relative to the robot root, smoothly gated when lost."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative_position_w = soccer.data.root_pos_w - robot.data.root_pos_w
    relative_position_b = quat_apply_inverse(robot.data.root_quat_w, relative_position_w)
    ball_height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    gate = _ball_observation_gate(
        env,
        relative_position_b,
        ball_height,
        clamp_dist,
        ground_height,
        transition_width,
    )
    return relative_position_b * gate.unsqueeze(-1)


def soccer_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    clamp_dist: float = 3.0,
    ground_height: float = 0.15,
    transition_width: float = 0.3,
) -> torch.Tensor:
    """Ball world velocity expressed in the robot frame and gated when lost."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative_position_w = soccer.data.root_pos_w - robot.data.root_pos_w
    relative_position_b = quat_apply_inverse(robot.data.root_quat_w, relative_position_w)
    velocity_b = quat_apply_inverse(robot.data.root_quat_w, soccer.data.root_lin_vel_w)
    ball_height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    gate = _ball_observation_gate(
        env,
        relative_position_b,
        ball_height,
        clamp_dist,
        ground_height,
        transition_width,
    )
    return velocity_b * gate.unsqueeze(-1)


def soccer_height(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer: RigidObject = env.scene[soccer_cfg.name]
    return (soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def feet_face_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Concatenated left/right foot-face positions in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    pos_w, _, _ = _foot_face_kinematics(env, robot_cfg)
    rel_w = pos_w - robot.data.root_pos_w.unsqueeze(1)
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, rel_w).flatten(start_dim=1)


def feet_face_projected_gravity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gravity direction projected into both foot-face frames."""
    _, quat_w, _ = _foot_face_kinematics(env, robot_cfg)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=quat_w.dtype)
    gravity = gravity.expand(quat_w.shape[0], 2, 3)
    return quat_apply_inverse(quat_w, gravity).flatten(start_dim=1)


def feet_face_to_ball_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Vectors from each foot face to the ball, expressed in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    foot_pos_w, _, _ = _foot_face_kinematics(env, robot_cfg)
    rel_w = soccer.data.root_pos_w.unsqueeze(1) - foot_pos_w
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, rel_w).flatten(start_dim=1)


def feet_face_lin_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Concatenated left/right foot-face linear velocities in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    _, _, velocity_w = _foot_face_kinematics(env, robot_cfg)
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, velocity_w).flatten(start_dim=1)


def feet_ang_vel_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Concatenated left/right foot angular velocities in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    ids = [_body_id(robot, LEFT_FOOT_BODY), _body_id(robot, RIGHT_FOOT_BODY)]
    velocity_w = robot.data.body_link_ang_vel_w[:, ids]
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, velocity_w).flatten(start_dim=1)


def next_kick_foot_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-hot encoding of the expected next kicking foot."""
    state = get_juggle_state(env)
    result = torch.stack(
        [(state.next_kick_foot == 1).float(), (state.next_kick_foot == 2).float()], dim=-1
    )
    return result * state.ball_in_juggle_range.unsqueeze(-1)


def juggle_gate_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return get_juggle_state(env).ball_in_juggle_range.unsqueeze(-1)
