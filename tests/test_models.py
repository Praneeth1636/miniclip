import torch

from mimicplay.models.bc_policy import BCPolicy
from mimicplay.models.vla_policy import VLAPolicy


def test_bc_policy_forward_shape() -> None:
    obs_shape = (96, 96, 3)
    n_actions = 4
    model = BCPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=12)
    x = torch.zeros(2, 12, 96, 96)
    y = model(x)
    assert y.shape == (2, n_actions)


def test_vla_policy_forward_shape() -> None:
    obs_shape = (96, 96, 3)
    n_actions = 4
    model = VLAPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=12)
    x = torch.zeros(2, 12, 96, 96)
    lang = torch.zeros(2, 384)
    y = model(x, lang)
    assert y.shape == (2, n_actions)

