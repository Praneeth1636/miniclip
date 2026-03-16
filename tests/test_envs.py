from pathlib import Path

import numpy as np

from mimicplay.envs import make


def test_grid_collector_reset_and_step() -> None:
    env = make("grid_collector")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert "episode" in info
    assert obs.shape == env.observation_shape

    action_space = env.get_action_space()
    assert len(action_space) == 4

    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert isinstance(obs2, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step" in info2

