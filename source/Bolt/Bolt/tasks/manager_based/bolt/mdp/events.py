"""Soccer-specific reset and domain-randomization events."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, yaw_quat

from .juggle_state import reset_juggle_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_soccer_drop(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    forward_range: tuple[float, float] = (0.4, 0.7),
    lateral_range: tuple[float, float] = (-0.2, 0.2),
    height_range: tuple[float, float] = (0.6, 1.5),
    forward_velocity_range: tuple[float, float] = (-0.2, 0.2),
    lateral_velocity_range: tuple[float, float] = (-0.2, 0.2),
    vertical_velocity_range: tuple[float, float] = (-0.3, 0.3),
) -> None:
    """Drop the ball in front of the robot using the reference task distribution."""
    if len(env_ids) == 0:
        return

    robot: Articulation = env.scene[robot_cfg.name]
    soccer: RigidObject = env.scene[soccer_cfg.name]
    n = len(env_ids)
    device = env.device

    root_pos = robot.data.root_pos_w[env_ids]
    root_quat = robot.data.root_quat_w[env_ids]
    yaw = yaw_quat(root_quat)
    forward_unit = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, -1)
    right_unit = torch.tensor([0.0, -1.0, 0.0], device=device).expand(n, -1)
    forward = quat_apply(yaw, forward_unit)
    right = quat_apply(yaw, right_unit)

    def sample(value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        return torch.empty(n, device=device).uniform_(low, high)

    forward_distance = sample(forward_range)
    lateral_distance = sample(lateral_range)
    height = sample(height_range)
    ball_pos = root_pos + forward * forward_distance.unsqueeze(-1) + right * lateral_distance.unsqueeze(-1)
    ball_pos[:, 2] = env.scene.env_origins[env_ids, 2] + height

    # Keep the predicted landing point inside the same safe interception band
    # used by BeyondAmp_Mjlab instead of sampling unconstrained lateral speed.
    fall_time = torch.sqrt(2.0 * torch.clamp(height - 0.3, min=0.01) / 9.81)
    forward_velocity = sample(forward_velocity_range)
    lateral_velocity = sample(lateral_velocity_range)
    forward_velocity = torch.clamp(
        forward_velocity,
        min=(0.35 - forward_distance) / fall_time,
        max=(0.60 - forward_distance) / fall_time,
    )
    lateral_velocity = torch.clamp(
        lateral_velocity,
        min=(-0.15 - lateral_distance) / fall_time,
        max=(0.15 - lateral_distance) / fall_time,
    )
    linear_velocity = (
        forward * forward_velocity.unsqueeze(-1)
        + right * lateral_velocity.unsqueeze(-1)
    )
    linear_velocity[:, 2] = sample(vertical_velocity_range)

    root_state = soccer.data.default_root_state[env_ids].clone()
    root_state[:, :3] = ball_pos
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    root_state[:, 7:10] = linear_velocity
    root_state[:, 10:13] = 0.0
    soccer.write_root_state_to_sim(root_state, env_ids=env_ids)

    first_foot = torch.where(lateral_distance >= 0.0, 2, 1).long()
    reset_juggle_state(env, env_ids, first_foot=first_foot)


def reset_joints_around_nominal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    joint_pos_dict: dict[str, float],
    position_range: tuple[float, float] = (-0.1, 0.1),
    velocity_range: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Reset joints around the explicit soccer pose used by the reference task."""
    if len(env_ids) == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    cache = getattr(env, "_bolt_nominal_reset_joint_pos", None)
    if cache is None:
        cache = asset.data.default_joint_pos[0].clone()
        patterns = [(re.compile(pattern), value) for pattern, value in joint_pos_dict.items()]
        for joint_id, joint_name in enumerate(asset.joint_names):
            for pattern, value in patterns:
                if pattern.fullmatch(joint_name):
                    cache[joint_id] = value
                    break
        env._bolt_nominal_reset_joint_pos = cache

    joint_ids = asset_cfg.joint_ids
    nominal = cache[joint_ids].unsqueeze(0).expand(len(env_ids), -1)
    joint_pos = nominal + torch.empty_like(nominal).uniform_(*position_range)
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])
    joint_vel = torch.empty_like(joint_pos).uniform_(*velocity_range)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    position_distribution_params: tuple[float, float],
) -> None:
    """Add per-environment encoder-zero offsets to default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        selected = asset.data.default_joint_pos[env_ids]
    else:
        selected = asset.data.default_joint_pos[env_ids[:, None], joint_ids]
    low, high = position_distribution_params
    randomized = selected + torch.empty_like(selected).uniform_(low, high)
    limits = asset.data.soft_joint_pos_limits[env_ids]
    if not isinstance(joint_ids, slice):
        limits = limits[:, joint_ids]
    randomized.clamp_(limits[..., 0], limits[..., 1])

    if isinstance(joint_ids, slice):
        asset.data.default_joint_pos[env_ids] = randomized
    else:
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = randomized

    # JointPositionAction snapshots the default offset during construction.
    action_term = env.action_manager.get_term("joint_pos")
    action_term._offset[env_ids] = asset.data.default_joint_pos[env_ids, action_term._joint_ids]


def set_soccer_diagonal_inertia(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    soccer_cfg: SceneEntityCfg = SceneEntityCfg("soccer"),
    reference_mass: float = 0.43,
    reference_diagonal_inertia: float = 0.0035,
) -> None:
    """Match the reference ball inertia while preserving randomized mass scaling."""
    soccer: RigidObject = env.scene[soccer_cfg.name]
    if env_ids is None:
        physx_env_ids = torch.arange(env.num_envs, device="cpu")
    elif isinstance(env_ids, torch.Tensor):
        physx_env_ids = env_ids.cpu()
    else:
        physx_env_ids = torch.tensor(env_ids, dtype=torch.long, device="cpu")

    masses = soccer.root_physx_view.get_masses().reshape(-1)
    inertias = soccer.root_physx_view.get_inertias().clone()
    diagonal = masses[physx_env_ids] * (reference_diagonal_inertia / reference_mass)
    inertias[physx_env_ids] = 0.0
    inertias[physx_env_ids, 0] = diagonal
    inertias[physx_env_ids, 4] = diagonal
    inertias[physx_env_ids, 8] = diagonal
    soccer.root_physx_view.set_inertias(inertias, physx_env_ids)

    data_env_ids = physx_env_ids.to(soccer.data.default_inertia.device)
    soccer.data.default_inertia[data_env_ids] = inertias[physx_env_ids].to(
        soccer.data.default_inertia.device
    )


def randomize_virtual_foot_faces(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float] = (-0.01, 0.01),
    rotation_range: tuple[float, float] = (-0.02, 0.02),
) -> None:
    """Randomize the virtual MuJoCo foot-face sites used by observations and rewards.

    The imported USD does not retain MuJoCo sites, so IsaacLab reconstructs the
    two foot-face frames from their ankle links. These per-environment offsets
    reproduce the reference task's site position/orientation randomization.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)

    if not hasattr(env, "_bolt_foot_face_pos_offset"):
        default = torch.tensor((0.17, 0.0, -0.005), device=env.device)
        env._bolt_foot_face_pos_offset = default.view(1, 1, 3).repeat(env.num_envs, 2, 1)
        env._bolt_foot_face_quat_offset = torch.zeros(env.num_envs, 2, 4, device=env.device)
        env._bolt_foot_face_quat_offset[..., 0] = 1.0

    low, high = position_range
    default = torch.tensor((0.17, 0.0, -0.005), device=env.device)
    position_noise = torch.empty(len(env_ids), 2, 3, device=env.device).uniform_(low, high)
    env._bolt_foot_face_pos_offset[env_ids] = default.view(1, 1, 3) + position_noise

    low, high = rotation_range
    angles = torch.empty(len(env_ids), 2, 3, device=env.device).uniform_(low, high)
    env._bolt_foot_face_quat_offset[env_ids] = quat_from_euler_xyz(
        angles[..., 0], angles[..., 1], angles[..., 2]
    )
