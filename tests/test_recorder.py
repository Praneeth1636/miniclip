from pathlib import Path

import h5py
import numpy as np

from mimicplay.data.recorder import _save_episode


def test_save_episode_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "demo_001.hdf5"
    frames = np.zeros((5, 96, 96, 3), dtype=np.uint8)
    actions = np.zeros((5,), dtype=np.int64)
    rewards = np.zeros((5,), dtype=np.float32)
    timestamps = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    metadata = {"env_name": "grid_collector", "player_id": "test", "date": "now", "success": True}

    _save_episode(
        path=path,
        frames=frames,
        actions=actions,
        rewards=rewards,
        timestamps=timestamps,
        language="collect all coins",
        metadata=metadata,
    )

    assert path.exists()
    with h5py.File(path, "r") as f:
        ep = f["episode"]
        assert ep["observations"].shape[0] == 5
        assert ep.attrs["language"] == "collect all coins"

