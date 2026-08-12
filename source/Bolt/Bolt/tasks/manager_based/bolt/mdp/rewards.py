"""Soccer-juggling rewards ported from BeyondAmp_Mjlab to IsaacLab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .juggle_state import LEFT_FOOT_BODY, RIGHT_FOOT_BODY, _body_id, get_juggle_state
from .observations import _foot_face_kinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return get_juggle_state(env).ball_in_juggle_range


def _feet_and_ball(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    use_face: bool = False,
) -> tuple[Articulation, RigidObject, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    if use_face:
        feet_pos, _, _ = _foot_face_kinematics(env, robot_cfg)
        left_pos, right_pos = feet_pos[:, 0], feet_pos[:, 1]
    else:
        left_id = _body_id(robot, LEFT_FOOT_BODY)
        right_id = _body_id(robot, RIGHT_FOOT_BODY)
        left_pos = robot.data.body_pos_w[:, left_id]
        right_pos = robot.data.body_pos_w[:, right_id]
    return robot, soccer, left_pos, right_pos, soccer.data.root_pos_w


def _joint_ids(robot: Articulation, names: tuple[str, ...]) -> list[int]:
    cache_name = "_bolt_joint_ids_" + "_".join(names)
    cached = getattr(robot, cache_name, None)
    if cached is not None:
        return cached
    ids: list[int] = []
    for name in names:
        matched, _ = robot.find_joints(name)
        ids.extend(int(value) for value in matched)
    setattr(robot, cache_name, ids)
    return ids


def foot_face_ball_distance_directional(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    std: float = 0.15,
    unreachable_height: float = 0.3,
    unreachable_vz: float = 0.2,
    kick_zone_max_height: float = 0.4,
) -> torch.Tensor:
    """Guide only the next expected foot face toward a reachable ball."""
    state = get_juggle_state(env)
    _, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    left_dist = torch.linalg.vector_norm(left_pos - ball_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_pos - ball_pos, dim=-1)
    distance = torch.where(state.next_kick_foot == 1, left_dist, right_dist)
    reward = torch.exp(-torch.square(distance) / (2.0 * std**2))
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    ball_vz = soccer.data.root_lin_vel_w[:, 2]
    unreachable = ((ball_vz > unreachable_vz) & (ball_height > unreachable_height)) | (
        ball_height > kick_zone_max_height
    )
    return reward * (state.hold_steps < 10).float() * (~unreachable).float() * state.ball_in_juggle_range


def robot_facing_ball_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    min_ball_dist: float = 0.4,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative_xy = soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.linalg.vector_norm(relative_xy, dim=-1)
    ball_direction = relative_xy / distance.unsqueeze(-1).clamp(min=1.0e-3)
    forward_unit = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    forward = quat_apply(yaw_quat(robot.data.root_quat_w), forward_unit)[:, :2]
    reward = ((forward * ball_direction).sum(dim=-1) + 1.0) * 0.5
    return torch.where(distance < min_ball_dist, torch.zeros_like(reward), reward) * _gate(env)


def next_kick_foot_height_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    target_height: float = 0.3,
    max_track_height: float = 0.4,
    std: float = 0.12,
) -> torch.Tensor:
    state = get_juggle_state(env)
    robot, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(env, robot_cfg, soccer_cfg)
    left_height = left_pos[:, 2] - env.scene.env_origins[:, 2]
    right_height = right_pos[:, 2] - env.scene.env_origins[:, 2]
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    descending_and_low = (soccer.data.root_lin_vel_w[:, 2] < 0.0) & (ball_height < max_track_height)
    target = torch.where(
        descending_and_low,
        ball_height.clamp(min=0.07, max=max_track_height),
        torch.full_like(ball_height, target_height),
    )
    foot_height = torch.where(state.next_kick_foot == 1, left_height, right_height)
    return torch.exp(-torch.square(foot_height - target) / (2.0 * std**2)) * state.ball_in_juggle_range


def kick_reward(env: ManagerBasedRLEnv, same_foot_scale: float = 0.0) -> torch.Tensor:
    state = get_juggle_state(env)
    scale = torch.where(
        state.alternated_this_step,
        torch.ones(env.num_envs, device=env.device),
        torch.full((env.num_envs,), same_foot_scale, device=env.device),
    )
    return state.kick_this_step.float() * scale * state.ball_in_juggle_range


def alternating_kick_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = get_juggle_state(env)
    return state.alternated_this_step.float() * state.ball_in_juggle_range


def kick_quality_reward(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    target_vz: float = 3.2,
    std: float = 1.0,
    min_vz: float = 0.5,
    linear_blend: float = 0.3,
) -> torch.Tensor:
    state = get_juggle_state(env)
    soccer: RigidObject = env.scene[soccer_cfg.name]
    ball_vz = soccer.data.root_lin_vel_w[:, 2]
    gaussian = torch.exp(-torch.square(ball_vz - target_vz) / (2.0 * std**2))
    ramp = torch.clamp((ball_vz - min_vz) / (target_vz - min_vz + 1.0e-6), 0.0, 1.0)
    reward = (1.0 - linear_blend) * gaussian + linear_blend * ramp
    reward = torch.where(ball_vz > min_vz, reward, torch.zeros_like(reward))
    same_foot = state.kick_this_step & ~state.alternated_this_step
    foot_gate = state.alternated_this_step.float() + 0.2 * same_foot.float()
    return reward * foot_gate * state.ball_in_juggle_range


def juggle_streak_reward(env: ManagerBasedRLEnv, max_count: int = 5) -> torch.Tensor:
    state = get_juggle_state(env)
    return torch.clamp(state.alternating_count.float() / max(max_count, 1), 0.0, 1.0) * state.ball_in_juggle_range


def ball_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-torch.square(height - target_height) / (2.0 * std**2)) * _gate(env)


def ball_approach_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    activate_dist: float = 0.95,
    deactivate_dist: float = 1.5,
    max_approach_vel: float = 0.6,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative = soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    direction = relative / distance.unsqueeze(-1).clamp(min=1.0e-3)
    approach = (robot.data.root_link_lin_vel_w[:, :2] * direction).sum(dim=-1)
    reward = torch.clamp(approach / max_approach_vel, 0.0, 1.0)
    active = (distance > activate_dist) & (distance < deactivate_dist)
    return reward * active.float() * _gate(env)


def ball_upward_velocity_reward(
    env: ManagerBasedRLEnv,
    target_vel: float,
    std: float,
    max_vel: float,
    min_vel: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    soccer: RigidObject = env.scene[soccer_cfg.name]
    velocity = soccer.data.root_lin_vel_w[:, 2]
    reward = torch.exp(-torch.square(velocity - target_vel) / (2.0 * std**2))
    reward = torch.where(velocity > min_vel, reward, torch.zeros_like(reward))
    over_kick = torch.clamp((velocity - max_vel) / std, min=0.0, max=1.0)
    same_foot = state.kick_this_step & ~state.alternated_this_step
    foot_gate = state.alternated_this_step.float() + 0.2 * same_foot.float()
    return (reward - over_kick) * foot_gate * state.ball_in_juggle_range


def ball_horizontal_velocity_penalty(
    env: ManagerBasedRLEnv,
    max_horiz_vel: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer: RigidObject = env.scene[soccer_cfg.name]
    speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w[:, :2], dim=-1)
    return torch.clamp(speed / max_horiz_vel, 0.0, 1.0) * _gate(env)


def kick_force_penalty(
    env: ManagerBasedRLEnv,
    max_speed: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    soccer: RigidObject = env.scene[soccer_cfg.name]
    speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w, dim=-1)
    excess = torch.clamp((speed - max_speed) / max_speed, 0.0, 1.0)
    return excess * state.kick_this_step.float() * state.ball_in_juggle_range


def robot_upright_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    juggle_discount: float = 0.9,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    gate = _gate(env)
    return upright * ((1.0 - gate) + gate * juggle_discount)


def same_foot_kick_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = get_juggle_state(env)
    return (state.kick_this_step & ~state.alternated_this_step & (state.prev_kick_foot != 0)).float()


def wrong_foot_proximity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    sigma: float = 0.16,
    abs_threshold: float = 0.1,
    activate_height: float = 0.5,
) -> torch.Tensor:
    state = get_juggle_state(env)
    _, _, left_pos, right_pos, ball_pos = _feet_and_ball(env, robot_cfg, soccer_cfg)
    left_dist = torch.linalg.vector_norm(left_pos - ball_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_pos - ball_pos, dim=-1)
    wrong_dist = torch.where(state.next_kick_foot == 1, right_dist, left_dist)
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    proximity = torch.exp(-torch.square(wrong_dist) / (2.0 * sigma**2))
    active = (wrong_dist < abs_threshold) & (ball_height < activate_height)
    return proximity * active.float() * state.ball_in_juggle_range


def double_foot_proximity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    max_dist: float = 0.22,
) -> torch.Tensor:
    _, _, left_pos, right_pos, ball_pos = _feet_and_ball(env, robot_cfg, soccer_cfg)
    left_dist = torch.linalg.vector_norm(left_pos - ball_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_pos - ball_pos, dim=-1)
    return ((left_dist < max_dist) & (right_dist < max_dist)).float() * _gate(env)


def both_feet_clamp_penalty(env: ManagerBasedRLEnv, saturate_steps: int = 8) -> torch.Tensor:
    state = get_juggle_state(env)
    return torch.clamp(state.both_feet_clamp_steps.float() / max(saturate_steps, 1), 0.0, 1.0)


def ball_stationary_near_foot_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    sigma: float = 0.4,
    vel_threshold: float = 0.5,
) -> torch.Tensor:
    _, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(env, robot_cfg, soccer_cfg)
    distance = torch.minimum(
        torch.linalg.vector_norm(left_pos - ball_pos, dim=-1),
        torch.linalg.vector_norm(right_pos - ball_pos, dim=-1),
    )
    proximity = torch.exp(-torch.square(distance) / (2.0 * sigma**2))
    speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w, dim=-1)
    return proximity * torch.clamp(1.0 - speed / vel_threshold, 0.0, 1.0)


def ball_hold_duration_penalty(env: ManagerBasedRLEnv, max_steps: int = 12) -> torch.Tensor:
    return torch.clamp(get_juggle_state(env).hold_steps.float() / max(max_steps, 1), 0.0, 1.0)


def stable_standing_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_std: float = 0.3,
    ang_vel_std: float = 0.5,
    base_fraction: float = 0.2,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    linear = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    angular = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=-1)
    linear_reward = torch.exp(-torch.square(linear) / (2.0 * vel_std**2))
    angular_reward = torch.exp(-torch.square(angular) / (2.0 * ang_vel_std**2))
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    standing = 0.4 * linear_reward + 0.3 * angular_reward + 0.3 * upright
    gate = _gate(env)
    return standing * (base_fraction + (1.0 - base_fraction) * (1.0 - gate))


def torso_upright_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torso_body_name: str = "torso_link",
    base_fraction: float = 0.9,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    torso_id = _body_id(robot, torso_body_name)
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    gravity_torso = quat_apply_inverse(robot.data.body_quat_w[:, torso_id], gravity)
    upright = torch.clamp(-gravity_torso[:, 2], 0.0, 1.0)
    gate = _gate(env)
    return upright * (base_fraction + (1.0 - base_fraction) * (1.0 - gate))


def torso_backward_lean_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torso_body_name: str = "torso_link",
    juggle_scale: float = 1.0,
    standing_scale: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    torso_id = _body_id(robot, torso_body_name)
    forward = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    forward_w = quat_apply(robot.data.body_quat_w[:, torso_id], forward)
    penalty = torch.square(torch.clamp(forward_w[:, 2], min=0.0))
    gate = _gate(env)
    return penalty * (standing_scale * (1.0 - gate) + juggle_scale * gate)


def juggling_yaw_velocity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_yaw_vel: float = 0.5,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return (
        torch.clamp(torch.abs(robot.data.root_link_ang_vel_w[:, 2]) / max_yaw_vel, 0.0, 1.0)
        * _gate(env)
    )


def double_contact_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return get_juggle_state(env).double_contact_this_step.float()


def upper_body_joint_penalty(
    env: ManagerBasedRLEnv,
    joint_names: tuple[str, ...],
    std: float,
    juggle_discount: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _joint_ids(robot, joint_names)
    deviation = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    raw = 1.0 - torch.exp(-torch.mean(torch.square(deviation), dim=-1) / std**2)
    gate = _gate(env)
    return raw * ((1.0 - gate) + gate * juggle_discount)


def arm_action_rate_penalty(
    env: ManagerBasedRLEnv,
    joint_names: tuple[str, ...],
    juggle_discount: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _joint_ids(robot, joint_names)
    raw = torch.sum(torch.square(env.action_manager.action[:, ids] - env.action_manager.prev_action[:, ids]), dim=-1)
    gate = _gate(env)
    return torch.clamp(raw, 0.0, 1.0) * ((1.0 - gate) + gate * juggle_discount)


def arm_symmetry_penalty(
    env: ManagerBasedRLEnv,
    left_joints: tuple[str, ...],
    right_joints: tuple[str, ...],
    sign_flip: tuple[float, ...],
    juggle_discount: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    left = robot.data.joint_pos[:, _joint_ids(robot, left_joints)]
    right = robot.data.joint_pos[:, _joint_ids(robot, right_joints)]
    flip = torch.tensor(sign_flip, device=env.device, dtype=left.dtype)
    raw = torch.mean(torch.square(left - right * flip), dim=-1).clamp(0.0, 1.0)
    gate = _gate(env)
    return raw * ((1.0 - gate) + gate * juggle_discount)


def joint_group_velocity_penalty(
    env: ManagerBasedRLEnv,
    joint_names: tuple[str, ...],
    max_vel: float,
    juggle_discount: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    velocity = torch.abs(robot.data.joint_vel[:, _joint_ids(robot, joint_names)])
    raw = torch.mean(torch.clamp(velocity - max_vel, min=0.0) / max_vel, dim=-1).clamp(0.0, 1.0)
    gate = _gate(env)
    return raw * ((1.0 - gate) + gate * juggle_discount)


def juggling_lateral_leg_pose_penalty(
    env: ManagerBasedRLEnv,
    joint_names: tuple[str, ...],
    std: float,
    juggle_discount: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _joint_ids(robot, joint_names)
    deviation = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
    raw = 1.0 - torch.exp(-torch.mean(torch.square(deviation), dim=-1) / std**2)
    gate = _gate(env)
    return raw * ((1.0 - gate) + gate * juggle_discount)


def ball_drop_idle_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    max_height: float = 0.5,
    min_height: float = 0.15,
    min_approach_vel: float = 0.1,
) -> torch.Tensor:
    robot, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(env, robot_cfg, soccer_cfg)
    left_id = _body_id(robot, LEFT_FOOT_BODY)
    right_id = _body_id(robot, RIGHT_FOOT_BODY)
    left_delta = ball_pos - left_pos
    right_delta = ball_pos - right_pos
    left_dir = left_delta / torch.linalg.vector_norm(left_delta, dim=-1).unsqueeze(-1).clamp(min=1.0e-3)
    right_dir = right_delta / torch.linalg.vector_norm(right_delta, dim=-1).unsqueeze(-1).clamp(min=1.0e-3)
    left_approach = (robot.data.body_link_lin_vel_w[:, left_id] * left_dir).sum(dim=-1)
    right_approach = (robot.data.body_link_lin_vel_w[:, right_id] * right_dir).sum(dim=-1)
    best_approach = torch.maximum(left_approach, right_approach)
    height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    vertical_velocity = soccer.data.root_lin_vel_w[:, 2]
    active = (
        (_gate(env) > 0.5)
        & (vertical_velocity < 0.0)
        & (height > min_height)
        & (height < max_height)
        & (best_approach < min_approach_vel)
    )
    urgency = torch.clamp(-vertical_velocity / 1.5, 0.0, 1.0)
    deficit = 1.0 - torch.clamp(best_approach / (min_approach_vel + 1.0e-6), 0.0, 1.0)
    return urgency * deficit * active.float()


def leg_action_rate_penalty(
    env: ManagerBasedRLEnv,
    joint_names: tuple[str, ...],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ids = _joint_ids(robot, joint_names)
    delta = env.action_manager.action[:, ids] - env.action_manager.prev_action[:, ids]
    return torch.mean(torch.square(delta), dim=-1).clamp(0.0, 1.0) * (1.0 - _gate(env))


def ball_alive_bonus(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    horizontal_speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w[:, :2], dim=-1)
    height_shape = torch.exp(-torch.square(height - 0.6) / (2.0 * 0.24**2))
    in_band = ((height > 0.18) & (height < 1.2)).float()
    horizontal_control = torch.clamp(1.0 - horizontal_speed / 2.5, 0.0, 1.0)
    return _gate(env) * in_band * (0.7 * height_shape + 0.3 * horizontal_control)


def ball_grounding_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return get_juggle_state(env).ball_just_grounded.float()


def action_rate_2nd_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(get_juggle_state(env).action_jerk), dim=-1)
