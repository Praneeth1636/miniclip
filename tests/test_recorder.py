import tempfile
from pathlib import Path

import h5py
import numpy as np

from mimicplay.data.recorder import _save_episode


def test_save_and_load_episode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "demo_001.hdf5"
        T, H, W = 10, 96, 96
        frames = np.random.randint(0, 255, (T, H, W, 3), dtype=np.uint8)
        actions = np.random.randint(0, 4, (T,), dtype=np.int64)
        rewards = np.random.rand(T).astype(np.float32)
        timestamps = np.linspace(0, 1, T)

        _save_episode(
            path,
            frames=frames,
            actions=actions,
            rewards=rewards,
            timestamps=timestamps,
            language="collect all coins",
            metadata={"env_name": "grid_collector", "player_id": "test", "date": "2026-01-01", "success": True},
        )

        assert path.exists()
        with h5py.File(path, "r") as f:
            ep = f["episode"]
            assert ep["observations"].shape == (T, H, W, 3)
            assert ep["actions"].shape == (T,)
            assert ep.attrs["language"] == "collect all coins"
            assert ep["metadata"].attrs["success"] is True

