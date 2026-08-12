"""Shared per-environment state for the soccer-juggling reward terms.

The reference MJLab task updates these buffers in a custom environment class.
IsaacLab's manager-based environment does not need a subclass: the state is
updated lazily once per policy step and is shared by rewards, observations,
terminations, and the curriculum.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"


def _body_id(asset: Articulation, name: str) -> int:
    cache_name = f"_bolt_body_id_{name}"
    cached = getattr(asset, cache_name, None)
    if cached is not None:
        return cached
    ids, _ = asset.find_bodies(name)
    if len(ids) != 1:
        raise ValueError(f"Expected one robot body matching '{name}', found {len(ids)}.")
    body_id = int(ids[0])
    setattr(asset, cache_name, body_id)
    return body_id


def _filtered_contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Return maximum filtered contact force over the sensor history."""
    sensor = env.scene[sensor_name]
    force = sensor.data.force_matrix_w_history
    if force is None:
        force = sensor.data.force_matrix_w
    if force is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.vector_norm(force, dim=-1).flatten(start_dim=1).amax(dim=1)


def _new_state(env: ManagerBasedRLEnv) -> SimpleNamespace:
    n = env.num_envs
    device = env.device
    zeros_long = lambda: torch.zeros(n, dtype=torch.long, device=device)
    zeros_bool = lambda: torch.zeros(n, dtype=torch.bool, device=device)
    return SimpleNamespace(
        updated_step=-1,
        kick_count=zeros_long(),
        left_kick_count=zeros_long(),
        right_kick_count=zeros_long(),
        alternated_kick_count=zeros_long(),
        same_foot_kick_count=zeros_long(),
        last_kick_foot=zeros_long(),
        prev_kick_foot=zeros_long(),
        next_kick_foot=torch.ones(n, dtype=torch.long, device=device),
        alternating_count=zeros_long(),
        kick_cooldown=zeros_long(),
        hold_steps=zeros_long(),
        both_feet_clamp_steps=zeros_long(),
        kick_this_step=zeros_bool(),
        alternated_this_step=zeros_bool(),
        double_contact_this_step=zeros_bool(),
        ball_just_grounded=zeros_bool(),
        ball_in_juggle_range=torch.ones(n, device=device),
        ball_started_near=torch.ones(n, dtype=torch.bool, device=device),
        action_jerk=torch.zeros(n, 1, device=device),
        prev_prev_action=None,
    )


def get_juggle_state(env: ManagerBasedRLEnv, update: bool = True) -> SimpleNamespace:
    state = getattr(env, "_bolt_juggle_state", None)
    if state is None:
        state = _new_state(env)
        env._bolt_juggle_state = state
    if update and state.updated_step != env.common_step_counter:
        _update_state(env, state)
    return state


def reset_juggle_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    first_foot: torch.Tensor | None = None,
) -> None:
    """Clear episode-local juggling state for selected environments."""
    state = get_juggle_state(env, update=False)
    if first_foot is None:
        first_foot = torch.ones(len(env_ids), dtype=torch.long, device=env.device)
    for name in (
        "kick_count",
        "left_kick_count",
        "right_kick_count",
        "alternated_kick_count",
        "same_foot_kick_count",
        "prev_kick_foot",
        "alternating_count",
        "kick_cooldown",
        "hold_steps",
        "both_feet_clamp_steps",
    ):
        getattr(state, name)[env_ids] = 0
    for name in (
        "kick_this_step",
        "alternated_this_step",
        "double_contact_this_step",
        "ball_just_grounded",
    ):
        getattr(state, name)[env_ids] = False
    state.next_kick_foot[env_ids] = first_foot
    # As in BeyondAmp_Mjlab, preset the previous foot to the opposite side so
    # a correct first contact receives the alternating-kick reward.
    state.last_kick_foot[env_ids] = 3 - first_foot
    state.ball_in_juggle_range[env_ids] = 1.0
    state.ball_started_near[env_ids] = True
    if state.prev_prev_action is not None:
        state.prev_prev_action[env_ids] = 0.0
    state.updated_step = -1


def _update_state(env: ManagerBasedRLEnv, state: SimpleNamespace) -> None:
    robot: Articulation = env.scene["robot"]
    soccer: RigidObject = env.scene["soccer"]
    left_id = _body_id(robot, LEFT_FOOT_BODY)
    right_id = _body_id(robot, RIGHT_FOOT_BODY)

    left_pos = robot.data.body_pos_w[:, left_id]
    right_pos = robot.data.body_pos_w[:, right_id]
    ball_pos = soccer.data.root_pos_w
    left_dist = torch.linalg.vector_norm(left_pos - ball_pos, dim=-1)
    right_dist = torch.linalg.vector_norm(right_pos - ball_pos, dim=-1)
    min_dist = torch.minimum(left_dist, right_dist)
    ball_speed = torch.linalg.vector_norm(soccer.data.root_lin_vel_w, dim=-1)
    ball_height = ball_pos[:, 2] - env.scene.env_origins[:, 2]

    state.kick_this_step.zero_()
    state.alternated_this_step.zero_()
    state.double_contact_this_step.zero_()

    # The two sensors are filtered against the ball and retain all physics
    # substeps in their history, so short impacts are not missed at policy rate.
    left_force = _filtered_contact_force(env, "left_foot_ball_contact")
    right_force = _filtered_contact_force(env, "right_foot_ball_contact")
    left_kick = (
        (left_force > 3.0)
        & (left_dist < 0.24)
        & (ball_speed > 0.55)
        & (ball_height > 0.20)
    )
    right_kick = (
        (right_force > 3.0)
        & (right_dist < 0.24)
        & (ball_speed > 0.55)
        & (ball_height > 0.20)
    )

    cooldown_active = state.kick_cooldown > 0
    left_kick &= ~cooldown_active
    right_kick &= ~cooldown_active
    any_kick = left_kick | right_kick
    double_contact = left_kick & right_kick
    single_kick = any_kick & ~double_contact

    current_foot = left_kick.long() + 2 * right_kick.long()
    closest_foot = torch.where(left_dist < right_dist, 1, 2)
    current_foot = torch.where(double_contact, closest_foot, current_foot)
    previous_foot = state.last_kick_foot.clone()
    alternated = single_kick & (current_foot != previous_foot)
    same_foot = single_kick & ~alternated & (previous_foot != 0)

    state.kick_this_step |= single_kick
    state.alternated_this_step |= alternated
    state.double_contact_this_step |= double_contact
    state.prev_kick_foot = torch.where(single_kick, previous_foot, state.prev_kick_foot)
    state.last_kick_foot = torch.where(single_kick, current_foot, previous_foot)
    state.next_kick_foot = torch.where(single_kick, 3 - current_foot, state.next_kick_foot)
    state.kick_count += single_kick.long()
    state.left_kick_count += (left_kick & ~double_contact).long()
    state.right_kick_count += (right_kick & ~double_contact).long()
    state.alternated_kick_count += alternated.long()
    state.same_foot_kick_count += same_foot.long()
    state.alternating_count = torch.where(
        alternated,
        state.alternating_count + 1,
        torch.where(same_foot, torch.zeros_like(state.alternating_count), state.alternating_count),
    )

    # The reference task updates these counters at physics rate. This port is
    # evaluated once at policy rate, so counters advance by ``decimation`` to
    # preserve the same durations in seconds.
    physics_steps = max(int(env.cfg.decimation), 1)
    cooldown_steps = 25
    state.kick_cooldown = torch.where(
        any_kick,
        torch.full_like(state.kick_cooldown, cooldown_steps),
        torch.clamp(state.kick_cooldown - physics_steps, min=0),
    )
    state.kick_cooldown = torch.where(
        min_dist > 0.30, torch.zeros_like(state.kick_cooldown), state.kick_cooldown
    )

    holding = (min_dist < 0.20) & (ball_speed < 0.25)
    state.hold_steps = torch.where(
        holding,
        state.hold_steps + physics_steps,
        torch.clamp(state.hold_steps - physics_steps, min=0),
    )
    both_close = (
        (left_dist < 0.22)
        & (right_dist < 0.22)
        & (torch.abs(left_dist - right_dist) < 0.05)
        & (ball_speed < 0.40)
    )
    state.both_feet_clamp_steps = torch.where(
        both_close,
        state.both_feet_clamp_steps + physics_steps,
        torch.zeros_like(state.both_feet_clamp_steps),
    )

    ball_dist = torch.linalg.vector_norm(
        soccer.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
    )
    dist_gate = 1.0 - torch.clamp((ball_dist - 1.2) / 0.4, 0.0, 1.0)
    new_gate = dist_gate * (ball_height >= 0.15).float()
    state.ball_just_grounded |= (state.ball_in_juggle_range > 0.5) & (new_gate < 0.5)
    state.ball_in_juggle_range = new_gate

    action = env.action_manager.action
    previous_action = env.action_manager.prev_action
    if state.prev_prev_action is None or state.prev_prev_action.shape != action.shape:
        state.prev_prev_action = torch.zeros_like(action)
    state.action_jerk = action - 2.0 * previous_action + state.prev_prev_action
    state.prev_prev_action.copy_(previous_action)
    state.updated_step = env.common_step_counter
