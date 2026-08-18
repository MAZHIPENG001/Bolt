"""Termination terms for the two-phase soccer-juggling curriculum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, yaw_quat

from .juggle_state import get_juggle_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase_enabled(env: ManagerBasedRLEnv) -> bool:
    return int(getattr(env, "_bolt_curriculum_phase", 0)) >= 1


def soccer_ball_on_ground_phase1b(
    env: ManagerBasedRLEnv,
    ball_radius: float = 0.11,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < ball_radius + 0.03) & state.ball_started_near


def soccer_ball_too_far_phase1b(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    distance = torch.linalg.vector_norm(
        soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return (distance > max_distance) & state.ball_started_near


def soccer_ball_too_high_phase1b(
    env: ManagerBasedRLEnv,
    max_height: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height > max_height) & state.ball_started_near


def soccer_ball_behind_robot_phase1b(
    env: ManagerBasedRLEnv,
    min_forward_dist: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative = soccer.data.root_pos_w - robot.data.root_pos_w
    forward_unit = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    forward = quat_apply(yaw_quat(robot.data.root_quat_w), forward_unit)
    forward_distance = (relative * forward).sum(dim=-1)
    return (forward_distance < min_forward_dist) & state.ball_started_near


def single_foot_bias_terminate(
    env: ManagerBasedRLEnv,
    same_foot_threshold: int = 3,
) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Consecutive misuse is what matters.  The old cumulative counter stopped
    # terminating the exploit after an episode had already accumulated >3
    # valid alternating contacts.
    return state.same_foot_streak >= same_foot_threshold


def both_feet_clamp_terminate(env: ManagerBasedRLEnv, min_steps: int = 20) -> torch.Tensor:
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (state.both_feet_clamp_steps >= min_steps) & state.ball_started_near


def low_ball_trap_terminate(env: ManagerBasedRLEnv, min_steps: int = 40) -> torch.Tensor:
    """End a phase-1b episode when the ball stays in the low-bounce band."""
    state = get_juggle_state(env)
    if not _phase_enabled(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (state.low_ball_steps >= min_steps) & state.ball_started_near


def incomplete_juggle_cycle_terminate(
    env: ManagerBasedRLEnv,
    max_steps: int = 300,
    warmup_max_steps: int = 400,
) -> torch.Tensor:
    """End an episode when no valid launch occurs for too long.

    Phase 1a gets a slightly longer acquisition window, but is intentionally
    not exempt: otherwise standing still for the full episode is its easiest
    local optimum.  The timeout applies before the first kick and between kicks.
    """
    state = get_juggle_state(env)
    threshold = max_steps if _phase_enabled(env) else warmup_max_steps
    return (state.cycle_wait_steps >= threshold) & state.ball_started_near


def ball_lost(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_horizontal_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Backward-compatible aggregate ball-loss check."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    distance = torch.linalg.vector_norm(
        soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return (height < minimum_height) | (distance > maximum_horizontal_distance)
