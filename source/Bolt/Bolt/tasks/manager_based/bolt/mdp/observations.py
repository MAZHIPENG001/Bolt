"""Teacher observations for the soccer juggling task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def soccer_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Return the exact ball position relative to the robot root in the robot frame.

    This is a privileged teacher observation. The future vision policy must replace it
    with a camera-derived estimate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative_position_w = soccer.data.root_pos_w - robot.data.root_pos_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_position_w)


def soccer_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Return the exact ball velocity relative to the robot root in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    relative_velocity_w = soccer.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_velocity_w)


def soccer_height(
    env: ManagerBasedRLEnv,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
) -> torch.Tensor:
    """Return the ball height in each local environment frame."""
    soccer: RigidObject = env.scene[soccer_cfg.name]
    return (soccer.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)
