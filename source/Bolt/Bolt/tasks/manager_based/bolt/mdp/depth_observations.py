"""Depth-only soccer-ball observations for the deployable policy.

The detector deliberately does not read the soccer rigid object's state.  It
unprojects the torso camera depth image, removes the ground and points close to
the known robot kinematics, and estimates the centre of the remaining sphere.
The depth environment also routes ball-dependent rewards through this cached
estimate.  Simulator ground truth remains available only to the asymmetric
critic and termination checks, and never enters the deployable actor inputs.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    transform_points,
    unproject_depth,
)

from .observations import _foot_face_kinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _new_depth_state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    return SimpleNamespace(
        last_step=-1,
        center_w=torch.zeros(env.num_envs, 3, device=env.device),
        velocity_w=torch.zeros(env.num_envs, 3, device=env.device),
        valid=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        lost_frames=torch.full((env.num_envs,), 1000, dtype=torch.long, device=env.device),
        candidate_count=torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )


def _depth_ball_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    camera_cfg: SceneEntityCfg,
    ball_radius: float,
    min_range: float,
    max_range: float,
    min_points: int,
    self_filter_radius: float,
    voxel_size: float,
    max_lost_frames: int,
) -> SimpleNamespace:
    """Update and return the cached depth-derived ball state in world frame."""
    state = getattr(env, "_bolt_depth_ball_state", None)
    if state is None:
        state = _new_depth_state(env)
        env._bolt_depth_ball_state = state

    # Observation terms for position, velocity and feet-to-ball are evaluated
    # separately.  Only process the image once for each control step.
    step = int(env.common_step_counter)
    if state.last_step == step:
        return state
    state.last_step = step

    robot: Articulation = env.scene[robot_cfg.name]
    camera: TiledCamera = env.scene[camera_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        depth = depth.squeeze(-1)

    # Isaac Lab's orthogonal depth and intrinsic matrix use the ROS optical
    # camera frame (+Z forward, +X right, +Y down).
    points_c = unproject_depth(depth, camera.data.intrinsic_matrices)
    points_w = transform_points(points_c, camera.data.pos_w, camera.data.quat_w_ros)
    # ``unproject_depth`` stores points in width-major pixel order, so use the
    # already unprojected Z coordinate rather than flattening the HxW image in
    # row-major order.
    flat_depth = points_c[..., 2]

    ground_z = env.scene.env_origins[:, 2:3]
    height = points_w[..., 2] - ground_z
    candidate = (
        torch.isfinite(flat_depth)
        & torch.isfinite(points_w).all(dim=-1)
        & (flat_depth > min_range)
        & (flat_depth < max_range)
        # Ground returns are rejected while retaining the visible ball surface.
        & (height > 0.025)
        & (height < 3.25)
    )

    # Remove the robot using forward kinematics, which is available from joint
    # encoders on the real system as well.  Iterating over links avoids forming
    # a large (environment, pixel, body, xyz) temporary tensor.
    radius_sq = self_filter_radius * self_filter_radius
    for body_id in range(robot.data.body_pos_w.shape[1]):
        delta = points_w - robot.data.body_pos_w[:, body_id].unsqueeze(1)
        candidate &= torch.sum(delta * delta, dim=-1) > radius_sq

    # Recover image neighbourhoods from Isaac Lab's width-major point order and
    # estimate an outward-facing surface normal with central differences.
    height_px, width_px = depth.shape[1:3]
    points_image_w = points_w.reshape(env.num_envs, width_px, height_px, 3).permute(0, 2, 1, 3)
    candidate_image = candidate.reshape(env.num_envs, width_px, height_px).permute(0, 2, 1)
    center_surface_w = points_image_w[:, 1:-1, 1:-1]
    tangent_u = points_image_w[:, 1:-1, 2:] - points_image_w[:, 1:-1, :-2]
    tangent_v = points_image_w[:, 2:, 1:-1] - points_image_w[:, :-2, 1:-1]
    normal_w = torch.linalg.cross(tangent_u, tangent_v, dim=-1)
    normal_length = torch.linalg.vector_norm(normal_w, dim=-1)
    normal_w = torch.nn.functional.normalize(normal_w, dim=-1, eps=1.0e-8)
    to_camera_w = camera.data.pos_w[:, None, None, :] - center_surface_w
    normal_w = torch.where(
        (torch.sum(normal_w * to_camera_w, dim=-1) >= 0.0).unsqueeze(-1),
        normal_w,
        -normal_w,
    )

    normal_valid = (
        candidate_image[:, 1:-1, 1:-1]
        & candidate_image[:, 1:-1, 2:]
        & candidate_image[:, 1:-1, :-2]
        & candidate_image[:, 2:, 1:-1]
        & candidate_image[:, :-2, 1:-1]
        & torch.isfinite(normal_length)
        & (normal_length > 1.0e-7)
        # Reject normals spanning a depth discontinuity at a silhouette edge.
        & (torch.linalg.vector_norm(tangent_u, dim=-1) < 0.20)
        & (torch.linalg.vector_norm(tangent_v, dim=-1) < 0.20)
    )
    center_votes_w = center_surface_w - ball_radius * normal_w
    center_votes_w = center_votes_w.reshape(env.num_envs, -1, 3)
    vote_valid = normal_valid.reshape(env.num_envs, -1)

    # Sphere votes must also lie outside the known robot volume.  Joint-state
    # forward kinematics supplies this self mask without ball ground truth.
    center_self_radius_sq = (self_filter_radius + 0.03) ** 2
    for body_id in range(robot.data.body_pos_w.shape[1]):
        delta = center_votes_w - robot.data.body_pos_w[:, body_id].unsqueeze(1)
        vote_valid &= torch.sum(delta * delta, dim=-1) > center_self_radius_sq

    # Quantized 3-D Hough voting: curved ball pixels agree on one centre voxel;
    # planar robot pixels cast dispersed votes.  Coordinates use the robot frame
    # horizontally and world height vertically to remain stable while balancing.
    relative_votes_w = center_votes_w - robot.data.root_pos_w.unsqueeze(1)
    root_rotation_w = matrix_from_quat(robot.data.root_quat_w)
    center_votes_b = torch.matmul(root_rotation_w.transpose(1, 2), relative_votes_w.transpose(1, 2))
    center_votes_b = center_votes_b.transpose(1, 2)
    vote_x = center_votes_b[..., 0]
    vote_y = center_votes_b[..., 1]
    vote_z = center_votes_w[..., 2] - ground_z

    x_min, x_max = -0.5, 4.0
    y_min, y_max = -2.0, 2.0
    z_min, z_max = 0.08, 3.25
    nx = math.ceil((x_max - x_min) / voxel_size)
    ny = math.ceil((y_max - y_min) / voxel_size)
    nz = math.ceil((z_max - z_min) / voxel_size)
    bin_x = torch.floor((vote_x - x_min) / voxel_size).long()
    bin_y = torch.floor((vote_y - y_min) / voxel_size).long()
    bin_z = torch.floor((vote_z - z_min) / voxel_size).long()
    vote_valid &= (
        (bin_x >= 0)
        & (bin_x < nx)
        & (bin_y >= 0)
        & (bin_y < ny)
        & (bin_z >= 0)
        & (bin_z < nz)
    )
    voxel_index = (bin_x.clamp(0, nx - 1) * ny + bin_y.clamp(0, ny - 1)) * nz
    voxel_index += bin_z.clamp(0, nz - 1)
    voxel_count = nx * ny * nz
    env_offset = torch.arange(env.num_envs, device=env.device).unsqueeze(-1) * voxel_count
    global_index = voxel_index + env_offset
    counts = torch.zeros(env.num_envs * voxel_count, dtype=torch.int32, device=env.device)
    counts.scatter_add_(0, global_index.flatten(), vote_valid.to(torch.int32).flatten())
    counts = counts.reshape(env.num_envs, voxel_count)
    best_count, best_voxel = counts.max(dim=-1)
    best_x = best_voxel // (ny * nz)
    best_y = (best_voxel // nz) % ny
    best_z = best_voxel % nz

    # Include immediate neighbour voxels when averaging so quantization
    # boundaries do not bias the final centre estimate.
    winning_votes = (
        vote_valid
        & ((bin_x - best_x.unsqueeze(-1)).abs() <= 1)
        & ((bin_y - best_y.unsqueeze(-1)).abs() <= 1)
        & ((bin_z - best_z.unsqueeze(-1)).abs() <= 1)
    )
    winning_count = winning_votes.sum(dim=-1)
    state.candidate_count = winning_count
    finite_votes_w = torch.where(
        winning_votes.unsqueeze(-1), center_votes_w, torch.zeros_like(center_votes_w)
    )
    measured_center_w = finite_votes_w.sum(dim=1) / winning_count.clamp_min(1).unsqueeze(-1)
    detected = (best_count >= min_points) & (winning_count >= min_points)

    # Observation-manager resets can evaluate terms before the first new image.
    # Never carry a position or finite-difference velocity across episodes.
    reset = env.episode_length_buf == 0
    previous_valid = state.valid & ~reset
    dt = float(env.step_dt)
    measured_velocity_w = (measured_center_w - state.center_w) / max(dt, 1.0e-6)
    measured_velocity_w = torch.clamp(measured_velocity_w, min=-30.0, max=30.0)
    velocity_valid = detected & previous_valid

    state.center_w = torch.where(detected.unsqueeze(-1), measured_center_w, state.center_w)
    state.velocity_w = torch.where(
        velocity_valid.unsqueeze(-1),
        measured_velocity_w,
        torch.where(detected.unsqueeze(-1), torch.zeros_like(state.velocity_w), 0.5 * state.velocity_w),
    )
    state.lost_frames = torch.where(
        detected,
        torch.zeros_like(state.lost_frames),
        state.lost_frames + 1,
    )
    state.valid = state.lost_frames <= max_lost_frames
    state.valid &= ~reset
    state.center_w = torch.where(reset.unsqueeze(-1), torch.zeros_like(state.center_w), state.center_w)
    state.velocity_w = torch.where(reset.unsqueeze(-1), torch.zeros_like(state.velocity_w), state.velocity_w)
    return state


def _depth_ball_state_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    ball_radius: float = 0.11,
    # Reject the robot chest and arms immediately in front of the torso camera.
    min_range: float = 0.35,
    max_range: float = 4.0,
    min_points: int = 6,
    self_filter_radius: float = 0.20,
    voxel_size: float = 0.08,
    max_lost_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[robot_cfg.name]
    state = _depth_ball_state(
        env,
        robot_cfg,
        camera_cfg,
        ball_radius,
        min_range,
        max_range,
        min_points,
        self_filter_radius,
        voxel_size,
        max_lost_frames,
    )
    relative_w = state.center_w - robot.data.root_pos_w
    position_b = quat_apply_inverse(robot.data.root_quat_w, relative_w)
    velocity_b = quat_apply_inverse(robot.data.root_quat_w, state.velocity_w)
    gate = state.valid.unsqueeze(-1)
    return position_b * gate, velocity_b * gate, gate


def depth_soccer_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
) -> torch.Tensor:
    """Depth-derived ball position in the robot root frame."""
    position_b, _, _ = _depth_ball_state_b(env, robot_cfg, camera_cfg)
    return position_b


def depth_soccer_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
) -> torch.Tensor:
    """Finite-difference velocity of the depth-derived ball centre."""
    _, velocity_b, _ = _depth_ball_state_b(env, robot_cfg, camera_cfg)
    return velocity_b


def depth_feet_face_to_ball_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
) -> torch.Tensor:
    """Vectors from both foot faces to the depth-derived ball centre."""
    robot: Articulation = env.scene[robot_cfg.name]
    ball_pos_b, _, valid = _depth_ball_state_b(env, robot_cfg, camera_cfg)
    foot_pos_w, _, _ = _foot_face_kinematics(env, robot_cfg)
    foot_rel_w = foot_pos_w - robot.data.root_pos_w.unsqueeze(1)
    root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    foot_pos_b = quat_apply_inverse(root_quat, foot_rel_w)
    result = ball_pos_b.unsqueeze(1) - foot_pos_b
    return (result * valid.unsqueeze(1)).flatten(start_dim=1)


def next_kick_foot_command_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Expected foot command from the shared perception-only reward state."""
    # Runtime import avoids a module cycle: the depth state itself uses the
    # detector implemented above.
    from .depth_juggle_state import get_depth_juggle_state

    state = get_depth_juggle_state(env)
    command = torch.stack(
        [
            (state.next_kick_foot == 1).float(),
            (state.next_kick_foot == 2).float(),
        ],
        dim=-1,
    )
    return command * state.perception_valid.unsqueeze(-1)
