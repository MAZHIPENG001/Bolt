"""Fail-fast checks for deployable depth-policy training and playback."""


DEPTH_TASK_NAME = "Bolt-Soccer-Depth-v0"
DEPTH_IMAGE_TASK_NAME = "Bolt-Soccer-DepthImage-v0"


def _network_inputs(model_cfg: dict) -> set[str]:
    return {str(layer.get("input")) for layer in model_cfg.get("network", [])}


def validate_depth_policy_config(task_name: str | None, agent_cfg: dict) -> None:
    """Ensure the depth actor cannot be wired to privileged critic states."""
    if not task_name or task_name.split(":")[-1] not in {DEPTH_TASK_NAME, DEPTH_IMAGE_TASK_NAME}:
        return

    models = agent_cfg.get("models", {})
    policy_inputs = _network_inputs(models.get("policy", {}))
    value_inputs = _network_inputs(models.get("value", {}))
    if not policy_inputs or policy_inputs != {"OBSERVATIONS"}:
        raise ValueError(
            f"depth visual actor must use only OBSERVATIONS; got {sorted(policy_inputs)}"
        )
    if not value_inputs or value_inputs != {"STATES"}:
        raise ValueError(
            f"depth visual critic must use STATES; got {sorted(value_inputs)}"
        )
    if not agent_cfg.get("agent", {}).get("observation_preprocessor"):
        raise ValueError("depth visual actor must configure an observation preprocessor")


def validate_depth_environment_config(task_name: str | None, env_cfg) -> None:
    """Ensure depth-task rewards cannot silently fall back to simulator truth."""
    if not task_name or task_name.split(":")[-1] != DEPTH_TASK_NAME:
        return
    if not bool(getattr(env_cfg, "depth_reward_inputs", False)):
        raise ValueError(f"{DEPTH_TASK_NAME} must enable depth_reward_inputs")
    if type(getattr(env_cfg, "rewards", None)).__name__ != "DepthRewardsCfg":
        raise ValueError(f"{DEPTH_TASK_NAME} must use DepthRewardsCfg")
