"""Check depth-only ball estimates against simulator truth for camera calibration."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=20, help="Steps to run; 0 runs until Isaac Sim is closed.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--show_depth",
    action="store_true",
    help="Show a live normalized depth image in an Isaac Sim window.",
)
parser.add_argument(
    "--camera_id",
    type=int,
    default=0,
    help="Environment/camera index displayed by --show_depth.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.show_depth and args_cli.headless:
    parser.error("--show_depth requires the GUI; remove --headless")
if not 0 <= args_cli.camera_id < args_cli.num_envs:
    parser.error("--camera_id must be in [0, --num_envs)")
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.ui as ui
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils import parse_env_cfg

import Bolt.tasks  # noqa: F401
from Bolt.tasks.manager_based.bolt import mdp


class DepthPreviewWindow:
    """Small Isaac Sim window displaying one normalized depth image."""

    def __init__(self, camera_id: int, near: float = 0.12, far: float = 4.0):
        self.camera_id = camera_id
        self.near = near
        self.far = far
        self.window = ui.Window("Bolt Depth Camera", width=640, height=520)
        self.provider = ui.ByteImageProvider()
        with self.window.frame:
            with ui.VStack(spacing=6):
                self.status = ui.Label(
                    f"env_{camera_id} | near={near:.2f} m | far={far:.2f} m | near pixels are bright",
                    height=24,
                )
                ui.ImageWithProvider(
                    self.provider,
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                )

    def update(self, depth: torch.Tensor) -> None:
        depth = depth.squeeze(-1) if depth.ndim == 3 else depth
        valid = torch.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        normalized = 1.0 - torch.clamp((depth - self.near) / (self.far - self.near), 0.0, 1.0)
        gray = torch.where(valid, normalized, torch.zeros_like(normalized))
        gray = (255.0 * gray).to(dtype=torch.uint8)
        alpha = torch.full_like(gray, 255)
        rgba = torch.stack((gray, gray, gray, alpha), dim=-1).contiguous().cpu()
        height, width = gray.shape
        self.provider.set_bytes_data(rgba.flatten().tolist(), [width, height])
        if torch.any(valid):
            valid_depth = depth[valid]
            self.status.text = (
                f"env_{self.camera_id} | min={float(valid_depth.min()):.3f} m "
                f"max={float(valid_depth.max()):.3f} m | near pixels are bright"
            )


def main() -> None:
    env_cfg = parse_env_cfg(
        "Bolt-Soccer-Depth-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env = gym.make("Bolt-Soccer-Depth-v0", cfg=env_cfg)
    env.reset()
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    soccer = unwrapped.scene["soccer"]
    depth_camera = unwrapped.scene["depth_camera"]
    preview = DepthPreviewWindow(args_cli.camera_id) if args_cli.show_depth else None

    step = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            env.step(actions)
            estimated = mdp.depth_soccer_position_in_robot_frame(
                unwrapped,
                robot_cfg=SceneEntityCfg("robot"),
                camera_cfg=SceneEntityCfg("depth_camera"),
            )
            truth = quat_apply_inverse(
                robot.data.root_quat_w,
                soccer.data.root_pos_w - robot.data.root_pos_w,
            )
            error = torch.linalg.vector_norm(estimated - truth, dim=-1)
            detector = unwrapped._bolt_depth_ball_state
            if preview is not None:
                preview.update(depth_camera.data.output["distance_to_image_plane"][args_cli.camera_id])
            print(
                f"step={step:03d} "
                f"visible={bool(detector.valid[0])} "
                f"candidates={int(detector.candidate_count[0])} "
                f"estimate={estimated[0].tolist()} "
                f"truth={truth[0].tolist()} "
                f"error={float(error[0]):.4f} m",
                flush=True,
            )
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
