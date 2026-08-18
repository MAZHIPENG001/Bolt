"""Soccer-juggling rewards ported from BeyondAmp_Mjlab to IsaacLab."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .depth_juggle_state import get_depth_juggle_state
from .juggle_state import LEFT_FOOT_BODY, RIGHT_FOOT_BODY, _body_id, get_juggle_state
from .observations import _foot_face_kinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _uses_depth_reward_input(env: ManagerBasedRLEnv) -> bool:
    return bool(getattr(env.cfg, "depth_reward_inputs", False))


def _reward_state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    """Select the task-event state used by the active reward configuration."""
    if _uses_depth_reward_input(env):
        return get_depth_juggle_state(env)
    return get_juggle_state(env)


def _reward_ball(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> RigidObject | SimpleNamespace:
    """Return ball kinematics from depth or simulator truth as configured."""
    if _uses_depth_reward_input(env):
        # Updating the perception state also updates the cached depth detector.
        get_depth_juggle_state(env)
        detector = env._bolt_depth_ball_state
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=detector.center_w,
                root_lin_vel_w=detector.velocity_w,
            )
        )
    return env.scene[soccer_cfg.name]


def _gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _reward_state(env).ball_in_juggle_range


def _ball_measurement_valid(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mask stale depth estimates without changing the Teacher reward path."""
    if _uses_depth_reward_input(env):
        return _reward_state(env).perception_valid.float()
    return torch.ones(env.num_envs, device=env.device)


def _feet_and_ball(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    use_face: bool = False,
) -> tuple[
    Articulation,
    RigidObject | SimpleNamespace,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    robot: Articulation = env.scene[robot_cfg.name]
    soccer = _reward_ball(env, soccer_cfg)
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
    ball_radius: float = 0.11,
    vertical_std: float = 0.08,
) -> torch.Tensor:
    """Guide the expected foot below the ball without rewarding penetration."""
    state = _reward_state(env)
    _, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    next_pos = torch.where((state.next_kick_foot == 1).unsqueeze(-1), left_pos, right_pos)
    delta = ball_pos - next_pos
    horizontal_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    vertical_error = delta[:, 2] - ball_radius
    reward = torch.exp(-torch.square(horizontal_distance) / (2.0 * std**2))
    reward *= torch.exp(-torch.square(vertical_error) / (2.0 * vertical_std**2))
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    ball_vz = soccer.data.root_lin_vel_w[:, 2]
    unreachable = ((ball_vz > unreachable_vz) & (ball_height > unreachable_height)) | (
        ball_height > kick_zone_max_height
    )
    contact_window = ((state.kick_count == 0) | state.flight_ready) & (
        ball_vz <= unreachable_vz
    )
    return (
        reward
        * contact_window.float()
        * (state.hold_steps < 10).float()
        * (~unreachable).float()
        * state.ball_in_juggle_range
    )


def robot_facing_ball_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    min_ball_dist: float = 0.4,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    soccer = _reward_ball(env, soccer_cfg)
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
    ball_radius: float = 0.11,
    activate_vz: float = 0.3,
) -> torch.Tensor:
    state = _reward_state(env)
    _, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    left_height = left_pos[:, 2] - env.scene.env_origins[:, 2]
    right_height = right_pos[:, 2] - env.scene.env_origins[:, 2]
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    descending_and_reachable = (
        (soccer.data.root_lin_vel_w[:, 2] < activate_vz) & (ball_height < max_track_height)
    )
    contact_window = (state.kick_count == 0) | state.flight_ready
    target = (ball_height - ball_radius).clamp(min=0.07, max=target_height)
    foot_height = torch.where(state.next_kick_foot == 1, left_height, right_height)
    return (
        torch.exp(-torch.square(foot_height - target) / (2.0 * std**2))
        * descending_and_reachable.float()
        * contact_window.float()
        * state.ball_in_juggle_range
    )


def kick_reward(env: ManagerBasedRLEnv, same_foot_scale: float = 0.0) -> torch.Tensor:
    state = _reward_state(env)
    scale = torch.where(
        state.expected_kick_this_step,
        torch.ones(env.num_envs, device=env.device),
        torch.full((env.num_envs,), same_foot_scale, device=env.device),
    )
    return state.kick_this_step.float() * scale * state.ball_in_juggle_range


def alternating_kick_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    state = _reward_state(env)
    return state.alternated_this_step.float() * state.ball_in_juggle_range


def kick_quality_reward(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    target_vz: float = 3.2,
    std: float = 1.0,
    min_vz: float = 0.5,
    linear_blend: float = 0.3,
) -> torch.Tensor:
    state = _reward_state(env)
    soccer = _reward_ball(env, soccer_cfg)
    ball_vz = soccer.data.root_lin_vel_w[:, 2]
    gaussian = torch.exp(-torch.square(ball_vz - target_vz) / (2.0 * std**2))
    ramp = torch.clamp((ball_vz - min_vz) / (target_vz - min_vz + 1.0e-6), 0.0, 1.0)
    reward = (1.0 - linear_blend) * gaussian + linear_blend * ramp
    reward = torch.where(ball_vz > min_vz, reward, torch.zeros_like(reward))
    foot_gate = state.expected_kick_this_step.float() + 0.2 * state.wrong_foot_this_step.float()
    return reward * foot_gate * state.ball_in_juggle_range


def juggle_streak_reward(env: ManagerBasedRLEnv, max_count: int = 5) -> torch.Tensor:
    state = _reward_state(env)
    # Pay the streak bonus only on a new alternating contact.  Paying it every
    # frame lets the policy earn a streak once and then keep a low trapped ball
    # near one foot for a much larger return than continued juggling.
    return (
        torch.clamp(state.alternating_count.float() / max(max_count, 1), 0.0, 1.0)
        * state.alternated_this_step.float()
        * state.ball_in_juggle_range
    )


def flight_cycle_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pay once when a kick has produced a released up-and-down ball arc."""
    state = _reward_state(env)
    return state.flight_completed_this_step.float() * state.ball_in_juggle_range


def _predicted_next_foot_intercept(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    soccer_cfg: SceneEntityCfg,
    ball_radius: float,
    gravity: float,
    max_flight_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return horizontal miss distance, descending flight time, and validity."""
    state = _reward_state(env)
    _, soccer, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    next_pos = torch.where((state.next_kick_foot == 1).unsqueeze(-1), left_pos, right_pos)
    ball_vz = soccer.data.root_lin_vel_w[:, 2]
    contact_height = next_pos[:, 2] + ball_radius
    discriminant = torch.square(ball_vz) + 2.0 * gravity * (ball_pos[:, 2] - contact_height)
    flight_time = (ball_vz + torch.sqrt(torch.clamp(discriminant, min=0.0))) / gravity
    predicted_xy = ball_pos[:, :2] + soccer.data.root_lin_vel_w[:, :2] * flight_time.unsqueeze(-1)
    miss_distance = torch.linalg.vector_norm(predicted_xy - next_pos[:, :2], dim=-1)
    valid = (discriminant >= 0.0) & (flight_time > 0.0) & (flight_time < max_flight_time)
    return miss_distance, flight_time, valid


def kick_landing_target_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    ball_radius: float = 0.11,
    gravity: float = 9.81,
    max_flight_time: float = 1.2,
    std: float = 0.16,
    min_upward_velocity: float = 0.8,
) -> torch.Tensor:
    """Reward a kick whose ballistic landing point is over the next foot."""
    state = _reward_state(env)
    soccer = _reward_ball(env, soccer_cfg)
    miss_distance, _, valid = _predicted_next_foot_intercept(
        env, robot_cfg, soccer_cfg, ball_radius, gravity, max_flight_time
    )
    reward = torch.exp(-torch.square(miss_distance) / (2.0 * std**2))
    active = (
        state.expected_kick_this_step
        & (soccer.data.root_lin_vel_w[:, 2] > min_upward_velocity)
        & valid
    )
    return reward * active.float() * state.ball_in_juggle_range


def next_foot_interception_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    ball_radius: float = 0.11,
    gravity: float = 9.81,
    max_flight_time: float = 1.2,
    std: float = 0.14,
    min_flight_time: float = 0.05,
    min_clearance: float = 0.08,
) -> torch.Tensor:
    """Move the expected foot underneath the predicted descending ball path."""
    state = _reward_state(env)
    _, _, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    next_pos = torch.where((state.next_kick_foot == 1).unsqueeze(-1), left_pos, right_pos)
    miss_distance, flight_time, valid = _predicted_next_foot_intercept(
        env, robot_cfg, soccer_cfg, ball_radius, gravity, max_flight_time
    )
    reward = torch.exp(-torch.square(miss_distance) / (2.0 * std**2))
    airborne = ball_pos[:, 2] > next_pos[:, 2] + ball_radius + min_clearance
    active = (
        (state.kick_count > 0)
        & (flight_time > min_flight_time)
        & valid
        & airborne
        & ~state.kick_this_step
    )
    return reward * active.float() * state.ball_in_juggle_range


def ball_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer = _reward_ball(env, soccer_cfg)
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
    soccer = _reward_ball(env, soccer_cfg)
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
    state = _reward_state(env)
    soccer = _reward_ball(env, soccer_cfg)
    velocity = soccer.data.root_lin_vel_w[:, 2]
    reward = torch.exp(-torch.square(velocity - target_vel) / (2.0 * std**2))
    reward = torch.where(velocity > min_vel, reward, torch.zeros_like(reward))
    over_kick = torch.clamp((velocity - max_vel) / std, min=0.0, max=1.0)
    foot_gate = state.expected_kick_this_step.float() + 0.2 * state.wrong_foot_this_step.float()
    return (reward - over_kick) * foot_gate * state.ball_in_juggle_range


def ball_horizontal_velocity_penalty(
    env: ManagerBasedRLEnv,
    max_horiz_vel: float,
    free_horiz_vel: float = 0.0,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    soccer = _reward_ball(env, soccer_cfg)
    speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w[:, :2], dim=-1)
    scale = max(max_horiz_vel - free_horiz_vel, 1.0e-6)
    return torch.clamp((speed - free_horiz_vel) / scale, 0.0, 1.0) * _gate(env)


def kick_force_penalty(
    env: ManagerBasedRLEnv,
    max_speed: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = _reward_state(env)
    soccer = _reward_ball(env, soccer_cfg)
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
    return _reward_state(env).wrong_foot_this_step.float()


def invalid_contact_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize touches that neither launch the ball nor follow a full flight."""
    return _reward_state(env).invalid_contact_this_step.float()


def wrong_foot_proximity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    sigma: float = 0.16,
    abs_threshold: float = 0.1,
    activate_height: float = 0.5,
    ball_radius: float = 0.11,
    vertical_tolerance: float = 0.08,
) -> torch.Tensor:
    state = _reward_state(env)
    _, _, left_pos, right_pos, ball_pos = _feet_and_ball(
        env, robot_cfg, soccer_cfg, use_face=True
    )
    wrong_pos = torch.where((state.next_kick_foot == 1).unsqueeze(-1), right_pos, left_pos)
    wrong_delta = ball_pos - wrong_pos
    wrong_dist = torch.linalg.vector_norm(wrong_delta[:, :2], dim=-1)
    vertical_error = torch.abs(wrong_delta[:, 2] - ball_radius)
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]
    proximity = torch.exp(-torch.square(wrong_dist) / (2.0 * sigma**2))
    active = (
        (wrong_dist < abs_threshold)
        & (vertical_error < vertical_tolerance)
        & (ball_height < activate_height)
    )
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
    state = _reward_state(env)
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
    return (
        proximity
        * torch.clamp(1.0 - speed / vel_threshold, 0.0, 1.0)
        * _ball_measurement_valid(env)
    )


def ball_hold_duration_penalty(env: ManagerBasedRLEnv, max_steps: int = 12) -> torch.Tensor:
    return torch.clamp(_reward_state(env).hold_steps.float() / max(max_steps, 1), 0.0, 1.0)


def low_ball_duration_penalty(env: ManagerBasedRLEnv, max_steps: int = 40) -> torch.Tensor:
    """Penalize staying continuously in the low-bounce band after the first kick."""
    return torch.clamp(
        _reward_state(env).low_ball_steps.float() / max(max_steps, 1), 0.0, 1.0
    )


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


def torso_lean_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torso_body_name: str = "torso_link",
    forward_tolerance: float = 0.10,
    backward_tolerance: float = 0.15,
    juggle_scale: float = 1.0,
    standing_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize forward and backward torso lean outside a small dead band."""
    robot: Articulation = env.scene[robot_cfg.name]
    torso_id = _body_id(robot, torso_body_name)
    forward = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    forward_w = quat_apply(robot.data.body_quat_w[:, torso_id], forward)
    forward_lean = torch.clamp(
        (-forward_w[:, 2] - forward_tolerance) / max(1.0 - forward_tolerance, 1.0e-6),
        0.0,
        1.0,
    )
    backward_lean = torch.clamp(
        (forward_w[:, 2] - backward_tolerance) / max(1.0 - backward_tolerance, 1.0e-6),
        0.0,
        1.0,
    )
    penalty = torch.maximum(torch.square(forward_lean), torch.square(backward_lean))
    gate = _gate(env)
    return penalty * (standing_scale * (1.0 - gate) + juggle_scale * gate)


def base_height_below_target_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.90,
    margin: float = 0.25,
) -> torch.Tensor:
    """Discourage the persistent crouch that can place the torso over the ball."""
    robot: Articulation = env.scene[robot_cfg.name]
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    deficit = torch.clamp((target_height - height) / max(margin, 1.0e-6), 0.0, 1.0)
    return torch.square(deficit)


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
    return _reward_state(env).double_contact_this_step.float()


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
    soccer = _reward_ball(env, soccer_cfg)
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    horizontal_speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w[:, :2], dim=-1)
    height_shape = torch.exp(-torch.square(height - 0.6) / (2.0 * 0.24**2))
    # The old 0.18 m lower bound paid an alive bonus for the low-foot trap.
    in_band = ((height > 0.28) & (height < 1.2)).float()
    horizontal_control = torch.clamp(1.0 - horizontal_speed / 2.5, 0.0, 1.0)
    return _gate(env) * in_band * (0.7 * height_shape + 0.3 * horizontal_control)


def ball_grounding_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _reward_state(env).ball_just_grounded.float()


def action_rate_2nd_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(_reward_state(env).action_jerk), dim=-1)
