"""Two-phase reward curriculum matching the reference soccer task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .depth_juggle_state import get_depth_juggle_state
from .juggle_state import get_juggle_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _curriculum_state(env: ManagerBasedRLEnv, update: bool = True):
    """Use the same truth/depth event source as the configured rewards."""
    if bool(getattr(env.cfg, "depth_reward_inputs", False)):
        return get_depth_juggle_state(env, update=update)
    return get_juggle_state(env, update=update)


PHASE_1A_WEIGHTS: dict[str, float] = {
    "foot_ball_distance": 6.0,
    "robot_facing_ball": 1.0,
    "next_kick_foot_height": 5.0,
    "kick": 8.0,
    "alternating_kick": 12.0,
    "kick_quality": 2.0,
    "juggle_streak": 2.0,
    "flight_cycle": 8.0,
    "kick_landing_target": 4.0,
    "next_foot_interception": 1.0,
    "ball_height": 0.5,
    "ball_horiz_vel_penalty": -1.0,
    "kick_force_penalty": -1.0,
    "ball_alive_bonus": 1.0,
    "ball_upward_vel": 1.0,
    "ball_grounding_penalty": -0.5,
    "ball_approach_vel": 0.5,
    "ball_drop_idle_penalty": -1.0,
    "stable_standing": 5.0,
    "torso_upright": 5.0,
    "torso_lean_penalty": -5.0,
    "base_height_penalty": -3.0,
    "waist_pitch_penalty": -2.0,
    "leg_action_rate_penalty": -0.5,
    "robot_upright": 5.0,
    "same_foot_kick_penalty": -4.0,
    "invalid_contact_penalty": -5.0,
    "ball_stationary_near_foot": -2.0,
    "ball_hold_duration": -2.0,
    "low_ball_duration": -4.0,
    "double_contact_penalty": -2.0,
    "wrong_foot_proximity_penalty": -2.0,
    "double_foot_proximity_penalty": -0.2,
    "both_feet_clamp_penalty": -0.2,
    "upper_body_joint_penalty": -3.0,
    "arm_symmetry_penalty": -1.0,
    "ankle_pitch_vel_penalty": -1.0,
    "ankle_roll_vel_penalty": -0.2,
    "juggling_lateral_leg_pose_penalty": -0.2,
    "juggling_yaw_penalty": 0.0,
    "arm_action_rate_penalty": -1.0,
    "action_rate_l2": -0.15,
    "action_rate_2nd_l2": -0.05,
    "joint_limit": -20.0,
    "undesired_contacts": -2.0,
}


PHASE_1B_WEIGHTS: dict[str, float] = {
    # Contact preparation is dense, so keep it below the one-shot rewards for
    # launching and completing a flight.  Otherwise parking a foot under the
    # ball dominates the return.
    "foot_ball_distance": 25.0,
    "robot_facing_ball": 4.0,
    "next_kick_foot_height": 8.0,
    "kick": 60.0,
    "alternating_kick": 180.0,
    "kick_quality": 20.0,
    "juggle_streak": 100.0,
    "flight_cycle": 120.0,
    "kick_landing_target": 80.0,
    "next_foot_interception": 8.0,
    "ball_height": 20.0,
    "ball_horiz_vel_penalty": -40.0,
    "kick_force_penalty": -20.0,
    "ball_alive_bonus": 20.0,
    "ball_upward_vel": 30.0,
    "ball_grounding_penalty": -4.0,
    "ball_approach_vel": 5.0,
    "ball_drop_idle_penalty": -5.0,
    "stable_standing": 15.0,
    "torso_upright": 15.0,
    "torso_lean_penalty": -50.0,
    "base_height_penalty": -30.0,
    "waist_pitch_penalty": -12.0,
    "leg_action_rate_penalty": -0.5,
    "robot_upright": 12.0,
    "same_foot_kick_penalty": -100.0,
    "invalid_contact_penalty": -80.0,
    "ball_stationary_near_foot": -20.0,
    "ball_hold_duration": -30.0,
    "low_ball_duration": -50.0,
    "double_contact_penalty": -20.0,
    "wrong_foot_proximity_penalty": -40.0,
    "double_foot_proximity_penalty": -5.0,
    "both_feet_clamp_penalty": -5.0,
    "upper_body_joint_penalty": -8.0,
    "arm_symmetry_penalty": -5.0,
    "ankle_pitch_vel_penalty": -3.0,
    "ankle_roll_vel_penalty": -10.0,
    "juggling_lateral_leg_pose_penalty": -10.0,
    "juggling_yaw_penalty": -2.0,
    "arm_action_rate_penalty": -5.0,
    "action_rate_l2": -0.15,
    "action_rate_2nd_l2": 0.0,
    "joint_limit": -20.0,
    "undesired_contacts": -2.0,
}


def _apply_weights(env: ManagerBasedRLEnv, weights: dict[str, float]) -> None:
    for name, weight in weights.items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = weight
        env.reward_manager.set_term_cfg(name, cfg)


def phase1a_to_1b_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    alternated_kick_ema_threshold: float = 2.0,
    per_foot_kick_ema_threshold: float = 0.75,
    foot_balance_threshold: float = 0.6,
    episode_len_ema_threshold: float = 300.0,
    min_global_steps: int = 240_000,
    ema_alpha: float = 0.005,
    initial_phase: int = 0,
) -> dict[str, float] | None:
    """Advance from stability learning to full juggling when metrics are ready."""
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=env.device)
    if len(env_ids) == 0:
        return None

    if not hasattr(env, "_bolt_curriculum_phase"):
        env._bolt_curriculum_phase = int(initial_phase)
    if not hasattr(env, "_bolt_alt_kick_ema"):
        env._bolt_alt_kick_ema = 0.0
        env._bolt_left_kick_ema = 0.0
        env._bolt_right_kick_ema = 0.0
        env._bolt_episode_len_ema = 10.0
        _apply_weights(env, PHASE_1B_WEIGHTS if initial_phase >= 1 else PHASE_1A_WEIGHTS)

    state = _curriculum_state(env, update=False)
    batch = torch.stack(
        [
            state.alternated_kick_count[env_ids].float().mean(),
            state.left_kick_count[env_ids].float().mean(),
            state.right_kick_count[env_ids].float().mean(),
            env.episode_length_buf[env_ids].float().mean(),
        ]
    ).tolist()
    alpha = ema_alpha
    env._bolt_alt_kick_ema = (1.0 - alpha) * env._bolt_alt_kick_ema + alpha * batch[0]
    env._bolt_left_kick_ema = (1.0 - alpha) * env._bolt_left_kick_ema + alpha * batch[1]
    env._bolt_right_kick_ema = (1.0 - alpha) * env._bolt_right_kick_ema + alpha * batch[2]
    env._bolt_episode_len_ema = (1.0 - alpha) * env._bolt_episode_len_ema + alpha * batch[3]

    balance = min(env._bolt_left_kick_ema, env._bolt_right_kick_ema) / max(
        env._bolt_left_kick_ema, env._bolt_right_kick_ema, 1.0e-6
    )
    if env._bolt_curriculum_phase == 0:
        ready = (
            env.common_step_counter >= min_global_steps
            and env._bolt_alt_kick_ema >= alternated_kick_ema_threshold
            and min(env._bolt_left_kick_ema, env._bolt_right_kick_ema) >= per_foot_kick_ema_threshold
            and balance >= foot_balance_threshold
            and env._bolt_episode_len_ema >= episode_len_ema_threshold
        )
        if ready:
            env._bolt_curriculum_phase = 1
            _apply_weights(env, PHASE_1B_WEIGHTS)

    return {
        "phase": float(env._bolt_curriculum_phase),
        "alternated_kick_ema": env._bolt_alt_kick_ema,
        "left_kick_ema": env._bolt_left_kick_ema,
        "right_kick_ema": env._bolt_right_kick_ema,
        "foot_balance": balance,
        "episode_length_ema": env._bolt_episode_len_ema,
    }
