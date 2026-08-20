"""Static contract tests for the raw-depth-image soccer actor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/agents/skrl_ppo_depth_image_cfg.yaml"
ENV = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/depth_image_env_cfg.py"
OBSERVATIONS = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/mdp/depth_image_observations.py"
REGISTRATION = ROOT / "source/Bolt/Bolt/tasks/manager_based/bolt/__init__.py"
sys.path.insert(0, str(ROOT / "scripts/skrl"))

from policy_contract import validate_depth_policy_config  # noqa: E402


class DepthImagePolicyContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_cfg = yaml.safe_load(CONFIG.read_text())

    def test_task_has_an_independent_registration_and_config(self) -> None:
        registration = REGISTRATION.read_text()
        self.assertIn('id="Bolt-Soccer-DepthImage-v0"', registration)
        self.assertIn("skrl_ppo_depth_image_cfg.yaml", registration)

    def test_actor_uses_only_deployable_observations(self) -> None:
        validate_depth_policy_config("Bolt-Soccer-DepthImage-v0", self.agent_cfg)
        self.assertEqual(self.agent_cfg["models"]["policy"]["network"][0]["input"], "OBSERVATIONS")
        self.assertEqual(self.agent_cfg["models"]["value"]["network"][0]["input"], "STATES")

    def test_actor_observation_is_raw_depth_without_ball_detector_terms(self) -> None:
        source = ENV.read_text()
        self.assertIn("mdp.raw_depth_image", source)
        for detector_term in (
            "depth_soccer_position_in_robot_frame",
            "depth_soccer_velocity_in_robot_frame",
            "depth_feet_face_to_ball_b",
            "next_kick_foot_command_obs",
        ):
            self.assertNotIn(detector_term, source)

    def test_raw_depth_reader_has_no_scene_object_or_point_cloud_access(self) -> None:
        source = OBSERVATIONS.read_text()
        for forbidden in (
            'env.scene["soccer"]',
            "unproject_depth",
            "transform_points",
            "RigidObject",
            "depth_soccer",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn('camera.data.output["distance_to_image_plane"]', source)


if __name__ == "__main__":
    unittest.main()
