"""PyTorch dataset for demonstration episodes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from mimicplay.data.augmentation import apply_augmentations


@dataclass
class DemoIndexEntry:
    """Index entry mapping a global timestep to an episode and local timestep."""

    episode_idx: int
    t: int


class DemoDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Load demonstration frames, actions, and language from HDF5 files."""

    def __init__(
        self,
        demo_dir: Path,
        frame_stack: int = 4,
        augment: bool = True,
    ) -> None:
        self.demo_dir = demo_dir.resolve()
        self.frame_stack = frame_stack
        self.augment = augment

        self.episodes: List[Dict[str, object]] = self._load_episodes()
        self.index: List[DemoIndexEntry] = self._build_index()

    def _load_episodes(self) -> List[Dict[str, object]]:
        episodes: List[Dict[str, object]] = []
        for path in sorted(self.demo_dir.rglob("demo_*.hdf5")):
            with h5py.File(path, "r") as f:
                ep = f["episode"]
                length = ep["actions"].shape[0]
                episodes.append(
                    {
                        "path": path,
                        "length": length,
                    }
                )
        if not episodes:
            raise RuntimeError(f"No episodes found in {self.demo_dir}")
        return episodes

    def _build_index(self) -> List[DemoIndexEntry]:
        index: List[DemoIndexEntry] = []
        for ep_idx, ep in enumerate(self.episodes):
            length = int(ep["length"])  # type: ignore[arg-type]
            for t in range(length):
                index.append(DemoIndexEntry(episode_idx=ep_idx, t=t))
        return index

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:  # type: ignore[override]
        entry = self.index[idx]
        ep_meta = self.episodes[entry.episode_idx]
        path: Path = ep_meta["path"]  # type: ignore[assignment]

        with h5py.File(path, "r") as f:
            ep = f["episode"]
            obs_stack = self._get_stack(ep["observations"], entry.t)
            action = int(ep["actions"][entry.t])
            language: str = str(ep.attrs["language"])

        if self.augment:
            obs_stack = apply_augmentations(obs_stack)

        # Normalize to [0, 1] and transpose to (C, H, W)
        obs = torch.from_numpy(obs_stack).permute(2, 0, 1).float() / 255.0
        action_t = torch.tensor(action, dtype=torch.long)
        return obs, action_t, language

    def _get_stack(self, obs_dataset: h5py.Dataset, t: int) -> np.ndarray:
        """Stack last N frames along channel dimension."""
        frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        for i in range(self.frame_stack):
            frame_idx = max(0, t - self.frame_stack + 1 + i)
            frame = obs_dataset[frame_idx]  # (H, W, 3)
            frames.append(frame)
        stacked = np.concatenate(list(frames), axis=-1)  # (H, W, 3*N)
        return stacked


def compute_dataset_stats(demo_dir: Path) -> None:
    """Print dataset statistics to stdout."""
    demo_dir = demo_dir.resolve()
    files = sorted(demo_dir.rglob("demo_*.hdf5"))
    if not files:
        print(f"No demo_*.hdf5 files found under {demo_dir}")
        return

    num_episodes = 0
    total_frames = 0
    lengths: List[int] = []
    successes = 0

    for path in files:
        with h5py.File(path, "r") as f:
            ep = f["episode"]
            T = ep["actions"].shape[0]
            num_episodes += 1
            total_frames += T
            lengths.append(T)
            meta = ep["metadata"]
            success = bool(meta.attrs.get("success", False))
            if success:
                successes += 1

    avg_len = float(total_frames) / max(1, num_episodes)
    success_rate = 100.0 * successes / max(1, num_episodes)

    print(
        f"{num_episodes} episodes, {total_frames} frames, "
        f"avg length {avg_len:.1f} steps, {success_rate:.1f}% success rate"
    )

