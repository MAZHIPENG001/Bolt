# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments per process/GPU (not the global total in distributed runs).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help=(
        "Interval, in policy iterations, for printing training metrics and writing TensorBoard data. "
        "Set to 0 to disable metric logging."
    ),
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.log_interval < 0:
    parser.error("--log_interval must be greater than or equal to 0")
# Enable rendering for both video recording and the depth-observation task.
if args_cli.video or (
    args_cli.task
    and args_cli.task.split(":")[-1] in {"Bolt-Soccer-Depth-v0", "Bolt-Soccer-DepthImage-v0"}
):
    args_cli.enable_cameras = True


def _read_rank_env(name: str, default: int) -> int:
    """Read and validate an integer rank environment variable."""
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        parser.error(f"Environment variable {name} must be an integer")


# torchrun/JAX launchers expose ranks before Isaac Sim starts. Bind the simulator
# here as well as in main(), so Kit, PhysX and the RL tensors all select the same
# logical CUDA device from the beginning of the process.
if args_cli.ml_framework.startswith("torch"):
    distributed_rank = _read_rank_env("RANK", 0)
    distributed_local_rank = _read_rank_env("LOCAL_RANK", 0)
    distributed_world_size = _read_rank_env("WORLD_SIZE", 1)
else:
    distributed_rank = _read_rank_env("JAX_RANK", 0)
    distributed_local_rank = _read_rank_env("JAX_LOCAL_RANK", 0)
    distributed_world_size = _read_rank_env("JAX_WORLD_SIZE", 1)

if distributed_world_size > 1 and not args_cli.distributed:
    parser.error("A multi-process launcher was detected, but --distributed was not specified")

if args_cli.distributed:
    if distributed_world_size <= 1:
        parser.error("--distributed requires torchrun (or a JAX multi-process launcher) with WORLD_SIZE > 1")
    if not 0 <= distributed_rank < distributed_world_size:
        parser.error(f"Invalid distributed rank {distributed_rank} for world size {distributed_world_size}")

    local_world_size_name = "LOCAL_WORLD_SIZE" if args_cli.ml_framework.startswith("torch") else "JAX_LOCAL_WORLD_SIZE"
    distributed_local_world_size = _read_rank_env(local_world_size_name, distributed_world_size)
    if not 0 <= distributed_local_rank < distributed_local_world_size:
        parser.error(
            f"Invalid local rank {distributed_local_rank} for local world size {distributed_local_world_size}"
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        parser.error(
            "Do not set CUDA_VISIBLE_DEVICES for distributed Isaac Sim/Isaac Lab runs. "
            "Omniverse and CUDA can enumerate the masked devices differently, which may cause "
            "illegal CUDA memory accesses. Unset it and let LOCAL_RANK select cuda:0,1,..."
        )

    args_cli.device = f"cuda:{distributed_local_rank}"
    print(
        "[INFO][DistributedLauncher] "
        f"rank={distributed_rank}/{distributed_world_size} "
        f"local_rank={distributed_local_rank} "
        f"device={args_cli.device}",
        flush=True,
    )

    # Hydra otherwise lets all ranks write to the same output directory.
    if not any(argument.startswith("hydra.run.dir=") for argument in hydra_args):
        rank_env_name = "RANK" if args_cli.ml_framework.startswith("torch") else "JAX_RANK"
        hydra_args.append(
            f"hydra.run.dir=outputs/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}/"
            f"rank_${{oc.env:{rank_env_name},0}}"
        )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import random
import shlex
from collections.abc import Mapping
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from policy_contract import validate_depth_environment_config, validate_depth_policy_config

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import Bolt.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _broadcast_from_main_process(value):
    """Broadcast a small Python value from rank 0 in a torch distributed run."""
    if args_cli.ml_framework.startswith("torch") and skrl.config.torch.is_distributed:
        import torch
        import torch.distributed as dist

        values = [value if distributed_rank == 0 else None]
        dist.broadcast_object_list(
            values,
            src=0,
            device=torch.device(f"cuda:{distributed_local_rank}"),
        )
        return values[0]
    return value


def _install_tensorwise_skrl_parameter_broadcast() -> None:
    """Avoid skrl's pickle-based broadcast of CUDA model state dictionaries.

    skrl 2.0 broadcasts ``state_dict()`` with ``broadcast_object_list``. The
    serialized tensors retain rank 0's CUDA device tag, so deserializing them
    on another rank can trigger a cross-device CUDA copy. Broadcast each rank's
    local parameter/buffer tensor in place instead.
    """
    import torch
    import torch.distributed as dist
    from skrl.models.torch import Model

    @torch.no_grad()
    def broadcast_parameters(self, *, rank: int = 0) -> None:
        for name, tensor in self.state_dict().items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"Model state entry {name!r} is not a tensor")
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"Model state entry {name!r} is on {tensor.device}; NCCL requires one local CUDA tensor per rank"
                )
            if tensor.numel() == 0:
                continue
            if tensor.is_contiguous():
                dist.broadcast(tensor, src=rank)
            else:
                contiguous_tensor = tensor.contiguous()
                dist.broadcast(contiguous_tensor, src=rank)
                tensor.copy_(contiguous_tensor)

    Model.broadcast_parameters = broadcast_parameters


if args_cli.distributed and args_cli.ml_framework.startswith("torch"):
    _install_tensorwise_skrl_parameter_broadcast()
    if distributed_rank == 0:
        print("[INFO][DistributedLauncher] Using tensor-wise skrl model broadcast", flush=True)


def _as_float(value):
    """Convert a scalar tracked by skrl to a Python float."""
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _summarize_tracking_data(tracking_data: Mapping) -> dict[str, float]:
    """Apply the same reductions as skrl's TensorBoard writer."""
    summary = {}
    for tag, values in tracking_data.items():
        try:
            numeric_values = [_as_float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if not numeric_values:
            continue
        if tag.endswith("(min)"):
            summary[tag] = min(numeric_values)
        elif tag.endswith("(max)"):
            summary[tag] = max(numeric_values)
        else:
            summary[tag] = sum(numeric_values) / len(numeric_values)
    return summary


def _format_metric(value: float) -> str:
    absolute_value = abs(value)
    if absolute_value >= 10000 or (absolute_value != 0 and absolute_value < 0.001):
        return f"{value:.4e}"
    return f"{value:.6f}"


def _attach_console_metric_printer(agent, rollouts: int) -> None:
    """Print the same scalar values whenever skrl writes to TensorBoard."""
    original_write_tracking_data = agent.write_tracking_data

    def write_tracking_data(*args, **kwargs):
        timestep = kwargs.get("timestep", args[0] if args else 0)
        timesteps = kwargs.get("timesteps", args[1] if len(args) > 1 else 0)
        summary = _summarize_tracking_data(agent.tracking_data)

        result = original_write_tracking_data(*args, **kwargs)

        progress = 100.0 * timestep / timesteps if timesteps else 0.0
        iteration = math.ceil(timestep / rollouts) if rollouts else timestep
        print(
            f"\n[TRAIN] iteration={iteration}  step={timestep}/{timesteps}  progress={progress:.2f}%",
            flush=True,
        )
        for tag, value in sorted(summary.items()):
            print(f"[TRAIN]   {tag}: {_format_metric(value)}", flush=True)
        return result

    agent.write_tracking_data = write_tracking_data


def _iter_models(models, prefix=""):
    """Yield models from both single-agent and multi-agent model mappings."""
    if isinstance(models, Mapping):
        for name, value in models.items():
            qualified_name = f"{prefix}/{name}" if prefix else str(name)
            if isinstance(value, Mapping):
                yield from _iter_models(value, qualified_name)
            else:
                yield qualified_name, value


def _count_array_elements(value) -> int:
    """Count array elements in a nested JAX/Flax state without importing JAX."""
    if isinstance(value, Mapping):
        return sum(_count_array_elements(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_array_elements(item) for item in value)
    shape = getattr(value, "shape", None)
    if shape is None:
        return 0
    return math.prod(shape) if shape else 1


def _model_parameter_counts(model) -> tuple[int | None, int | None]:
    """Return total and trainable parameter counts for Torch or JAX models."""
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        model_parameters = list(parameters())
        total = sum(parameter.numel() for parameter in model_parameters)
        trainable = sum(parameter.numel() for parameter in model_parameters if parameter.requires_grad)
        return total, trainable

    state_dict = getattr(model, "state_dict", None)
    state = getattr(state_dict, "params", state_dict)
    total = _count_array_elements(state)
    return (total, total) if total else (None, None)


def _print_training_summary(env, runner, agent_cfg: dict, log_dir: str, rollouts: int) -> None:
    """Print the effective training setup and instantiated model sizes."""
    trainer_cfg = agent_cfg["trainer"]
    ppo_cfg = agent_cfg["agent"]
    timesteps = int(trainer_cfg["timesteps"])
    num_envs = int(getattr(env, "num_envs", 1))
    iterations = math.ceil(timesteps / rollouts)
    mini_batches = int(ppo_cfg.get("mini_batches", 1))
    samples_per_update = num_envs * rollouts
    global_num_envs = num_envs * distributed_world_size
    global_samples_per_update = samples_per_update * distributed_world_size
    mini_batch_size = samples_per_update // mini_batches
    write_interval = int(getattr(runner.agent, "write_interval", 0))

    print("\n[TRAIN] ==================== Training configuration ====================")
    print(f"[TRAIN] task: {args_cli.task}")
    print(f"[TRAIN] algorithm/framework: {args_cli.algorithm} / {args_cli.ml_framework}")
    print(f"[TRAIN] device: {getattr(env, 'device', 'unknown')}")
    print(f"[TRAIN] seed: {agent_cfg['seed']}")
    print(
        f"[TRAIN] parallel environments: {num_envs} per process/GPU, "
        f"{global_num_envs} global across {distributed_world_size} process(es)"
    )
    print(f"[TRAIN] observation space: {getattr(env, 'observation_space', 'unknown')}")
    print(f"[TRAIN] state space: {getattr(env, 'state_space', 'unknown')}")
    print(f"[TRAIN] action space: {getattr(env, 'action_space', 'unknown')}")
    print(f"[TRAIN] iterations / vector steps: {iterations} / {timesteps}")
    print(
        f"[TRAIN] rollouts / samples per update: {rollouts} / "
        f"{samples_per_update} local / {global_samples_per_update} global"
    )
    print(
        "[TRAIN] epochs / mini-batches / mini-batch size: "
        f"{ppo_cfg.get('learning_epochs')} / {mini_batches} / {mini_batch_size}"
    )
    print(f"[TRAIN] learning rate: {ppo_cfg.get('learning_rate')}")
    if write_interval:
        print(
            f"[TRAIN] metric interval: {write_interval} vector steps "
            f"(~{write_interval / rollouts:g} policy iterations)"
        )
    else:
        print("[TRAIN] metric logging: disabled")
    print(f"[TRAIN] experiment directory: {log_dir}")
    if args_cli.checkpoint:
        print(f"[TRAIN] resume checkpoint: {args_cli.checkpoint}")

    print("[TRAIN] models:")
    seen_models = {}
    for role, model in _iter_models(getattr(runner.agent, "models", {})):
        if model is None:
            continue
        if id(model) in seen_models:
            print(f"[TRAIN]   {role}: shared with {seen_models[id(model)]}")
            continue
        seen_models[id(model)] = role
        total, trainable = _model_parameter_counts(model)
        parameter_text = "parameter count unavailable"
        if total is not None:
            parameter_text = f"{total:,} parameters ({trainable:,} trainable)"
        print(f"[TRAIN]   {role}: {type(model).__name__}, {parameter_text}")
    print("[TRAIN] =================================================================\n")


def _flush_final_metrics(agent, timesteps: int) -> None:
    """Write a final partial logging window and flush the TensorBoard event file."""
    if getattr(agent, "write_interval", 0) > 0 and getattr(agent, "tracking_data", None):
        agent.write_tracking_data(timestep=timesteps, timesteps=timesteps)
    writer = getattr(agent, "writer", None)
    if writer is not None and hasattr(writer, "flush"):
        writer.flush()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    validate_depth_policy_config(args_cli.task, agent_cfg)
    validate_depth_environment_config(args_cli.task, env_cfg)
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    rollouts = int(agent_cfg["agent"]["rollouts"])
    agent_cfg["agent"]["experiment"]["write_interval"] = args_cli.log_interval * rollouts
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        sampled_seed = random.randint(0, 10000) if distributed_rank == 0 else None
        args_cli.seed = _broadcast_from_main_process(sampled_seed)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # skrl offsets its own seed by rank, but the environment is constructed before
    # Runner does so. Offset the environment seed explicitly to avoid identical
    # trajectories and domain randomization on every GPU.
    env_cfg.seed = agent_cfg["seed"] + distributed_rank if args_cli.distributed else agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    is_main_process = distributed_rank == 0
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if is_main_process and args_cli.log_interval:
        tensorboard_command = (
            f"conda run -n isaaclab tensorboard --logdir {shlex.quote(log_root_path)} --port 6006"
        )
        print(f"[INFO] View live training curves: {tensorboard_command}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    requested_log_dir = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
        if is_main_process
        else None
    )
    log_dir = _broadcast_from_main_process(requested_log_dir)
    if log_dir is None:
        # JAX does not use torch.distributed for object broadcasts. Its launcher
        # should start ranks closely enough for this naming fallback.
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    if is_main_process:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Surface simulator-side asynchronous CUDA failures before the first NCCL
    # model collective. Otherwise ProcessGroupNCCL reports the stale error and
    # makes a PhysX/Kit failure look like a communication failure.
    if args_cli.distributed and args_cli.ml_framework.startswith("torch"):
        import torch

        try:
            torch.cuda.synchronize(device=env_cfg.sim.device)
        except RuntimeError as error:
            raise RuntimeError(
                "CUDA failed while creating the Isaac Lab environment, before skrl model synchronization"
            ) from error

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    if is_main_process:
        _print_training_summary(env, runner, agent_cfg, log_dir, rollouts)
        if getattr(runner.agent, "write_interval", 0) > 0:
            _attach_console_metric_printer(runner.agent, rollouts)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()
    if is_main_process:
        _flush_final_metrics(runner.agent, int(agent_cfg["trainer"]["timesteps"]))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
