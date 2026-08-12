# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def ball_height_reward(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    target_height: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Reward lifting the ball from the ground up to the target juggling height."""
    soccer: RigidObject = env.scene[soccer_cfg.name]
    height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp((height - minimum_height) / (target_height - minimum_height), min=0.0, max=1.0)


def ball_centered_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Reward keeping the ball horizontally close to the robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    distance_xy = torch.linalg.vector_norm(
        soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return torch.exp(-torch.square(distance_xy / std))


def feet_to_ball_proximity_exp(
    env: ManagerBasedRLEnv,
    std: float,
    ball_radius: float,
    foot_face_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Reward bringing either foot's front contact surface close to the ball."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]

    foot_positions_w = robot.data.body_pos_w[:, robot_cfg.body_ids]
    foot_quaternions_w = robot.data.body_quat_w[:, robot_cfg.body_ids]
    offset = torch.tensor(foot_face_offset, device=env.device, dtype=foot_positions_w.dtype)
    offset = offset.expand_as(foot_positions_w)
    foot_face_positions_w = foot_positions_w + quat_apply(foot_quaternions_w, offset)

    center_distance = torch.linalg.vector_norm(
        foot_face_positions_w - soccer.data.root_pos_w.unsqueeze(1), dim=-1
    )
    surface_distance = torch.clamp(center_distance - ball_radius, min=0.0)
    closest_distance = torch.min(surface_distance, dim=1).values
    return torch.exp(-torch.square(closest_distance / std))


def ball_upward_velocity(
    env: ManagerBasedRLEnv,
    max_velocity: float,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Reward upward ball velocity created by a kick."""
    soccer: RigidObject = env.scene[soccer_cfg.name]
    return torch.clamp(soccer.data.root_lin_vel_w[:, 2], min=0.0, max=max_velocity) / max_velocity
