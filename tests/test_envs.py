import numpy as np
import pytest

from mimicplay.envs import make


ENV_NAMES = ["grid_collector", "dodge_runner", "build_bridge"]


@pytest.fixture(params=ENV_NAMES)
def env(request):
    return make(request.param)


def test_reset_returns_correct_shape(env) -> None:
    obs, info = env.reset()
    assert obs.shape == env.observation_shape
    assert isinstance(info, dict)


def test_step_returns_correct_types(env) -> None:
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space_nonempty(env) -> None:
    actions = env.get_action_space()
    assert len(actions) > 0
    assert all(isinstance(a, str) for a in actions)


def test_render_matches_observation_shape(env) -> None:
    env.reset()
    frame = env.render()
    assert frame.shape == env.observation_shape

