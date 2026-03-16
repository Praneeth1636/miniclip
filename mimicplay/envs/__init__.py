"""Environment registry and utilities."""

from __future__ import annotations

from typing import Dict, Type

from mimicplay.envs.base import BaseEnv
from mimicplay.envs.grid_collector import GridCollectorEnv
from mimicplay.envs.dodge_runner import DodgeRunnerEnv
from mimicplay.envs.build_bridge import BuildBridgeEnv

EnvRegistry = Dict[str, Type[BaseEnv]]


_ENVS: EnvRegistry = {
    "grid_collector": GridCollectorEnv,
    "dodge_runner": DodgeRunnerEnv,
    "build_bridge": BuildBridgeEnv,
}


def make(env_name: str) -> BaseEnv:
    """Instantiate an environment by name.

    Args:
        env_name: Registered environment identifier.

    Returns:
        Instantiated environment.

    Raises:
        KeyError: If the environment name is unknown.
    """
    if env_name not in _ENVS:
        raise KeyError(f"Unknown environment '{env_name}'. Available: {sorted(_ENVS.keys())}")
    return _ENVS[env_name]()

