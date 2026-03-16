import torch

from mimicplay.models.bc_policy import BCPolicy


def test_bc_policy_forward_shape() -> None:
    obs_shape = (96, 96, 3)
    n_actions = 4
    model = BCPolicy(obs_shape=obs_shape, n_actions=n_actions)
    x = torch.zeros(2, 3, 96, 96)
    y = model(x)
    assert y.shape == (2, n_actions)

