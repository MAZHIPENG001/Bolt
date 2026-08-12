"""Termination terms for the soccer juggling task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_lost(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    maximum_horizontal_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Terminate if the ball falls below the configured height or travels too far from the robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    ball_height = soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    horizontal_distance = torch.linalg.vector_norm(
        soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    return torch.logical_or(ball_height < minimum_height, horizontal_distance > maximum_horizontal_distance)
