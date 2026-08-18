"""Static contract tests for the deployable depth-only soccer actor."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
DEPTH_CONFIG = (
    ROOT
    / "source/Bolt/Bolt/tasks/manager_based/bolt/agents/skrl_ppo_depth_cfg.yaml"
)
DEPTH_ENV = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/depth_env_cfg.py"
DEPTH_OBSERVATIONS = (
    ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/depth_observations.py"
)
DEPTH_JUGGLE_STATE = (
    ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/depth_juggle_state.py"
)
REWARDS = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/rewards.py"
CURRICULUM = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/curriculum.py"
EVENTS = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/events.py"
REGISTRATION = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/__init__.py"
sys.path.insert(0, str(ROOT / "scripts/skrl"))

from policy_contract import (  # noqa: E402
    validate_depth_environment_config,
    validate_depth_policy_config,
)


class DepthPolicyContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_cfg = yaml.safe_load(DEPTH_CONFIG.read_text())

    def test_depth_task_uses_dedicated_actor_config(self) -> None:
        registration = REGISTRATION.read_text()
        self.assertIn('id="Bolt-Soccer-Depth-v0"', registration)
        self.assertIn("skrl_ppo_depth_cfg.yaml", registration)

    def test_actor_uses_policy_observations_and_critic_uses_states(self) -> None:
        validate_depth_policy_config("Bolt-Soccer-Depth-v0", self.agent_cfg)
        policy_input = self.agent_cfg["models"]["policy"]["network"][0]["input"]
        value_input = self.agent_cfg["models"]["value"]["network"][0]["input"]
        self.assertEqual(policy_input, "OBSERVATIONS")
        self.assertEqual(value_input, "STATES")

    def test_validator_rejects_privileged_actor_input(self) -> None:
        self.agent_cfg["models"]["policy"]["network"][0]["input"] = "STATES"
        with self.assertRaisesRegex(ValueError, "must use only OBSERVATIONS"):
            validate_depth_policy_config("Bolt-Soccer-Depth-v0", self.agent_cfg)

    def test_validator_rejects_truth_reward_fallback(self) -> None:
        class DepthRewardsCfg:
            pass

        env_cfg = type(
            "DepthEnvCfgStub",
            (),
            {"depth_reward_inputs": True, "rewards": DepthRewardsCfg()},
        )()
        validate_depth_environment_config("Bolt-Soccer-Depth-v0", env_cfg)
        env_cfg.depth_reward_inputs = False
        with self.assertRaisesRegex(ValueError, "must enable depth_reward_inputs"):
            validate_depth_environment_config("Bolt-Soccer-Depth-v0", env_cfg)

    def test_all_teacher_ball_terms_are_overridden_by_depth_terms(self) -> None:
        source = DEPTH_ENV.read_text()
        for function_name in (
            "depth_soccer_position_in_robot_frame",
            "depth_soccer_velocity_in_robot_frame",
            "next_kick_foot_command_obs",
            "depth_feet_face_to_ball_b",
        ):
            self.assertIn(f"mdp.{function_name}", source)

    def test_depth_environment_overrides_reward_input_only(self) -> None:
        tree = ast.parse(DEPTH_ENV.read_text())
        rewards_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DepthRewardsCfg"
        )
        self.assertEqual(ast.unparse(rewards_class.bases[0]), "RewardsCfg")
        depth_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BoltDepthEnvCfg"
        )
        self.assertEqual(ast.unparse(depth_class.bases[0]), "BoltEnvCfg")
        assigned_names = {
            target.id
            for node in depth_class.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.assertIn("rewards", assigned_names)
        self.assertIn("depth_reward_inputs", assigned_names)
        self.assertTrue({"actions", "events", "terminations", "curriculum"}.isdisjoint(assigned_names))

    def test_actor_perception_code_has_no_ball_truth_or_filtered_contact(self) -> None:
        source = DEPTH_OBSERVATIONS.read_text()
        forbidden_fragments = (
            'env.scene["soccer"]',
            "env.scene['soccer']",
            "RigidObject",
            "get_juggle_state",
            "_filtered_contact_force",
            "left_foot_ball_contact",
            "right_foot_ball_contact",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, source)

    def test_depth_reward_event_state_has_no_ball_truth_or_filtered_contact(self) -> None:
        source = DEPTH_JUGGLE_STATE.read_text()
        forbidden_fragments = (
            'env.scene["soccer"]',
            "env.scene['soccer']",
            "RigidObject",
            "get_juggle_state",
            "_filtered_contact_force",
            "left_foot_ball_contact",
            "right_foot_ball_contact",
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, source)

    def test_all_reward_ball_access_is_routed_by_environment_contract(self) -> None:
        source = REWARDS.read_text()
        # The only simulator-ball and truth-state accesses are the explicit
        # Teacher fallbacks inside the two routing helpers.
        self.assertEqual(source.count("return get_juggle_state(env)"), 1)
        self.assertEqual(source.count("return env.scene[soccer_cfg.name]"), 1)
        self.assertIn("return get_depth_juggle_state(env)", source)
        self.assertNotIn("soccer: RigidObject = env.scene[soccer_cfg.name]", source)

    def test_depth_curriculum_uses_depth_reward_state(self) -> None:
        source = CURRICULUM.read_text()
        self.assertIn('getattr(env.cfg, "depth_reward_inputs", False)', source)
        self.assertIn("get_depth_juggle_state(env, update=update)", source)

    def test_reset_invalidates_depth_state_without_sampled_first_foot(self) -> None:
        source = EVENTS.read_text()
        self.assertIn("reset_depth_juggle_state(env, env_ids)", source)
        self.assertNotIn("reset_depth_juggle_state(env, env_ids, first_foot", source)
        depth_source = DEPTH_JUGGLE_STATE.read_text()
        self.assertIn("detector.last_step = -1", depth_source)


if __name__ == "__main__":
    unittest.main()
