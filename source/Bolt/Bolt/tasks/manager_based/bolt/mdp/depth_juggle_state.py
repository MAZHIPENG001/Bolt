"""Depth-derived juggling events used by depth rewards and actor commands.

This module never reads the simulated ball rigid body or ball-filtered contact
sensors. Position and velocity come from the depth detector; kick/contact
events are inferred from ball-to-foot geometry and perceived launch motion.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation

from .depth_observations import _depth_ball_state_b
from .juggle_state import (
    MIN_CYCLE_APEX_HEIGHT,
    MIN_KICK_HEIGHT,
    MIN_KICK_UPWARD_VELOCITY,
    MIN_RELEASE_DISTANCE,
)
from .observations import _foot_face_kinematics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PERCEIVED_CONTACT_DISTANCE = 0.20
PERCEIVED_LAUNCH_DISTANCE = 0.35


def _new_depth_juggle_state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    n = env.num_envs
    device = env.device
    zeros_long = lambda: torch.zeros(n, dtype=torch.long, device=device)
    zeros_bool = lambda: torch.zeros(n, dtype=torch.bool, device=device)
    return SimpleNamespace(
        updated_step=-1,
        initialized=zeros_bool(),
        perception_valid=zeros_bool(),
        previous_left_near=zeros_bool(),
        previous_right_near=zeros_bool(),
        kick_count=zeros_long(),
        left_kick_count=zeros_long(),
        right_kick_count=zeros_long(),
        alternated_kick_count=zeros_long(),
        same_foot_kick_count=zeros_long(),
        last_kick_foot=zeros_long(),
        prev_kick_foot=zeros_long(),
        next_kick_foot=torch.ones(n, dtype=torch.long, device=device),
        alternating_count=zeros_long(),
        same_foot_streak=zeros_long(),
        kick_cooldown=zeros_long(),
        cycle_wait_steps=zeros_long(),
        hold_steps=zeros_long(),
        low_ball_steps=zeros_long(),
        both_feet_clamp_steps=zeros_long(),
        flight_peak_height=torch.zeros(n, device=device),
        flight_active=zeros_bool(),
        flight_released=zeros_bool(),
        flight_ready=zeros_bool(),
        kick_this_step=zeros_bool(),
        expected_kick_this_step=zeros_bool(),
        wrong_foot_this_step=zeros_bool(),
        alternated_this_step=zeros_bool(),
        invalid_contact_this_step=zeros_bool(),
        flight_completed_this_step=zeros_bool(),
        double_contact_this_step=zeros_bool(),
        ball_just_grounded=zeros_bool(),
        ball_in_juggle_range=torch.zeros(n, device=device),
        ball_started_near=zeros_bool(),
        action_jerk=torch.zeros(n, 1, device=device),
        prev_prev_action=None,
    )


def _reset_depth_juggle_state(state: SimpleNamespace, reset: torch.Tensor) -> None:
    for name in (
        "kick_count",
        "left_kick_count",
        "right_kick_count",
        "alternated_kick_count",
        "same_foot_kick_count",
        "last_kick_foot",
        "prev_kick_foot",
        "alternating_count",
        "same_foot_streak",
        "kick_cooldown",
        "cycle_wait_steps",
        "hold_steps",
        "low_ball_steps",
        "both_feet_clamp_steps",
    ):
        getattr(state, name)[reset] = 0
    for name in (
        "initialized",
        "perception_valid",
        "previous_left_near",
        "previous_right_near",
        "flight_active",
        "flight_released",
        "flight_ready",
        "kick_this_step",
        "expected_kick_this_step",
        "wrong_foot_this_step",
        "alternated_this_step",
        "invalid_contact_this_step",
        "flight_completed_this_step",
        "double_contact_this_step",
        "ball_just_grounded",
        "ball_started_near",
    ):
        getattr(state, name)[reset] = False
    state.next_kick_foot[reset] = 1
    state.flight_peak_height[reset] = 0.0
    state.ball_in_juggle_range[reset] = 0.0
    if state.prev_prev_action is not None:
        state.prev_prev_action[reset] = 0.0


def get_depth_juggle_state(
    env: ManagerBasedRLEnv, update: bool = True
) -> SimpleNamespace:
    """Return cached perception-only juggling state for the current step."""
    state = getattr(env, "_bolt_depth_juggle_state", None)
    if state is None:
        state = _new_depth_juggle_state(env)
        env._bolt_depth_juggle_state = state
    if update and state.updated_step != int(env.common_step_counter):
        _update_depth_juggle_state(env, state)
    return state


def reset_depth_juggle_state(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
) -> None:
    """Clear perception state after selected environments are reset."""
    state = get_depth_juggle_state(env, update=False)
    reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset[env_ids] = True
    _reset_depth_juggle_state(state, reset)
    state.updated_step = -1

    # Rewards are evaluated before reset observations in ManagerBasedRLEnv.
    # Invalidate the detector cache so the post-reset observation rerenders and
    # cannot reuse the previous episode's perceived ball state.
    detector = getattr(env, "_bolt_depth_ball_state", None)
    if detector is not None:
        detector.center_w[env_ids] = 0.0
        detector.velocity_w[env_ids] = 0.0
        detector.valid[env_ids] = False
        detector.lost_frames[env_ids] = 1000
        detector.candidate_count[env_ids] = 0
        detector.last_step = -1


def _update_depth_juggle_state(
    env: ManagerBasedRLEnv, state: SimpleNamespace
) -> None:
    robot: Articulation = env.scene["robot"]
    ball_pos_b, _, valid_column = _depth_ball_state_b(env)
    valid = valid_column.squeeze(-1)
    detector = env._bolt_depth_ball_state
    ball_pos = detector.center_w
    ball_velocity = detector.velocity_w

    foot_pos, _, _ = _foot_face_kinematics(env)
    foot_distance = torch.linalg.vector_norm(ball_pos.unsqueeze(1) - foot_pos, dim=-1)
    left_dist, right_dist = foot_distance[:, 0], foot_distance[:, 1]
    min_dist = foot_distance.amin(dim=-1)
    closest_foot = torch.argmin(foot_distance, dim=-1) + 1
    ball_speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
    ball_vz = ball_velocity[:, 2]
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]

    reset = env.episode_length_buf == 0
    if torch.any(reset):
        _reset_depth_juggle_state(state, reset)
    state.perception_valid = valid & ~reset

    initialize = state.perception_valid & ~state.initialized
    state.next_kick_foot = torch.where(initialize, closest_foot, state.next_kick_foot)
    state.initialized |= initialize

    for name in (
        "kick_this_step",
        "expected_kick_this_step",
        "wrong_foot_this_step",
        "alternated_this_step",
        "invalid_contact_this_step",
        "flight_completed_this_step",
        "double_contact_this_step",
        "ball_just_grounded",
    ):
        getattr(state, name).zero_()

    physics_steps = max(int(env.cfg.decimation), 1)
    left_near = state.perception_valid & (left_dist <= PERCEIVED_CONTACT_DISTANCE)
    right_near = state.perception_valid & (right_dist <= PERCEIVED_CONTACT_DISTANCE)
    left_onset = left_near & ~state.previous_left_near
    right_onset = right_near & ~state.previous_right_near
    state.previous_left_near = left_near
    state.previous_right_near = right_near
    double_contact = left_onset & right_onset
    proximity_onset = left_onset | right_onset

    active_flight = state.flight_active & state.perception_valid
    state.flight_peak_height = torch.where(
        active_flight,
        torch.maximum(state.flight_peak_height, ball_height),
        state.flight_peak_height,
    )
    state.flight_released |= active_flight & (min_dist > MIN_RELEASE_DISTANCE)
    completed_flight = (
        active_flight
        & state.flight_released
        & (state.flight_peak_height >= MIN_CYCLE_APEX_HEIGHT)
        & (ball_vz <= 0.0)
    )
    newly_completed = completed_flight & ~state.flight_ready
    state.flight_completed_this_step |= newly_completed
    state.flight_ready |= completed_flight

    launch_contact = (
        state.perception_valid
        & (state.kick_cooldown == 0)
        & (ball_vz >= MIN_KICK_UPWARD_VELOCITY)
        & (ball_height >= MIN_KICK_HEIGHT)
        & (min_dist <= PERCEIVED_LAUNCH_DISTANCE)
        & ~double_contact
    )
    had_previous_kick = state.kick_count > 0
    cycle_available = ~had_previous_kick | state.flight_ready
    valid_kick = launch_contact & cycle_available
    current_foot = closest_foot
    expected_kick = valid_kick & (current_foot == state.next_kick_foot)
    wrong_foot = valid_kick & ~expected_kick
    previous_foot = state.last_kick_foot.clone()
    alternated = expected_kick & had_previous_kick

    state.kick_this_step |= valid_kick
    state.expected_kick_this_step |= expected_kick
    state.wrong_foot_this_step |= wrong_foot
    state.alternated_this_step |= alternated
    state.invalid_contact_this_step |= proximity_onset & ~valid_kick
    state.double_contact_this_step |= double_contact
    state.prev_kick_foot = torch.where(valid_kick, previous_foot, state.prev_kick_foot)
    state.last_kick_foot = torch.where(valid_kick, current_foot, previous_foot)
    state.next_kick_foot = torch.where(valid_kick, 3 - current_foot, state.next_kick_foot)
    state.kick_count += valid_kick.long()
    state.left_kick_count += (valid_kick & (current_foot == 1)).long()
    state.right_kick_count += (valid_kick & (current_foot == 2)).long()
    state.alternated_kick_count += alternated.long()
    state.same_foot_kick_count += wrong_foot.long()
    state.alternating_count = torch.where(
        alternated,
        state.alternating_count + 1,
        torch.where(wrong_foot, torch.zeros_like(state.alternating_count), state.alternating_count),
    )
    state.same_foot_streak = torch.where(
        wrong_foot,
        state.same_foot_streak + 1,
        torch.where(alternated, torch.zeros_like(state.same_foot_streak), state.same_foot_streak),
    )

    state.flight_active = torch.where(
        launch_contact, torch.ones_like(state.flight_active), state.flight_active
    )
    state.flight_released = torch.where(
        launch_contact, torch.zeros_like(state.flight_released), state.flight_released
    )
    state.flight_ready = torch.where(
        launch_contact, torch.zeros_like(state.flight_ready), state.flight_ready
    )
    state.flight_peak_height = torch.where(
        launch_contact, ball_height, state.flight_peak_height
    )
    state.kick_cooldown = torch.where(
        launch_contact,
        torch.full_like(state.kick_cooldown, 25),
        torch.clamp(state.kick_cooldown - physics_steps, min=0),
    )

    holding = (left_near | right_near) & (ball_speed < 0.35)
    holding |= state.perception_valid & (min_dist < 0.20) & (ball_speed < 0.25)
    state.hold_steps = torch.where(
        holding,
        state.hold_steps + physics_steps,
        torch.clamp(state.hold_steps - physics_steps, min=0),
    )
    low_ball = (
        state.perception_valid
        & (state.kick_count > 0)
        & (ball_height > 0.14)
        & (ball_height < 0.30)
    )
    state.low_ball_steps = torch.where(
        low_ball,
        state.low_ball_steps + physics_steps,
        torch.zeros_like(state.low_ball_steps),
    )
    state.cycle_wait_steps += physics_steps
    state.cycle_wait_steps = torch.where(
        valid_kick, torch.zeros_like(state.cycle_wait_steps), state.cycle_wait_steps
    )
    both_close = (
        state.perception_valid
        & (left_dist < 0.22)
        & (right_dist < 0.22)
        & (torch.abs(left_dist - right_dist) < 0.05)
        & (ball_speed < 0.40)
    )
    state.both_feet_clamp_steps = torch.where(
        both_close,
        state.both_feet_clamp_steps + physics_steps,
        torch.zeros_like(state.both_feet_clamp_steps),
    )

    ball_dist = torch.linalg.vector_norm(ball_pos_b[:, :2], dim=-1)
    dist_gate = 1.0 - torch.clamp((ball_dist - 1.2) / 0.4, 0.0, 1.0)
    new_gate = (
        dist_gate
        * (ball_height >= 0.15).float()
        * state.perception_valid.float()
    )
    state.ball_just_grounded |= (
        (state.ball_in_juggle_range > 0.5)
        & (new_gate < 0.5)
        & state.perception_valid
        & ~reset
    )
    state.ball_in_juggle_range = new_gate
    state.ball_started_near |= initialize & (ball_dist <= 1.6)

    action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    if state.prev_prev_action is None or state.prev_prev_action.shape != action.shape:
        state.prev_prev_action = torch.zeros_like(action)
    state.action_jerk = action - 2.0 * previous_action + state.prev_prev_action
    state.prev_prev_action.copy_(previous_action)
    state.updated_step = int(env.common_step_counter)
