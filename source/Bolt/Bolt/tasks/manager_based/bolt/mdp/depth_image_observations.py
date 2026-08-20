"""Raw depth-image observations for end-to-end visual soccer policies.

This module intentionally contains no ball segmentation, point-cloud
processing, or geometric ball-state estimation.  It only turns the camera's
depth buffer into a finite, normalized tensor suitable for an actor network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def raw_depth_image(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    min_depth: float = 0.12,
    max_depth: float = 4.0,
) -> torch.Tensor:
    """Return the torso camera's normalized depth image flattened per environment.

    Invalid sensor returns are encoded as the far clipping distance.  Valid
    depth values are linearly mapped to ``[-1, 1]``.  The pixel order and all
    scene information are otherwise left untouched, so the policy learns its
    own visual features directly from the image.
    """
    camera: TiledCamera = env.scene[camera_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        depth = depth.squeeze(-1)

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth = depth.clamp(min=min_depth, max=max_depth)
    normalized = 2.0 * (depth - min_depth) / (max_depth - min_depth) - 1.0
    return normalized.flatten(start_dim=1)
