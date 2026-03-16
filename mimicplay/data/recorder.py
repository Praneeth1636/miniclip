"""Demonstration recording and replay utilities."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pygame

from mimicplay.envs import make as make_env


def run_recording_session(env_name: str, task_instruction: str, player_id: str, output_root: Path) -> None:
    """Run an interactive recording session and save episodes as HDF5.

    Keys:
        R: start a new recording episode.
        Q: discard current episode.
        ESC: stop session.
    """
    output_root = output_root.resolve()
    env_dir = output_root / env_name / _sanitize_task(task_instruction)
    env_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(env_name)
    obs, info = env.reset()
    _ = (obs, info)

    recording = False
    frames: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    timestamps: List[float] = []
    t0 = time.time()
    episode_idx = _next_episode_index(env_dir)

    print("Recording controls: R=start, Q=discard, ESC=quit session.")

    clock = pygame.time.Clock()
    running = True
    step = 0
    while running:
        action = None
        reward = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    if not recording:
                        print("Starting new recording episode.")
                        recording = True
                        frames, actions, rewards, timestamps = [], [], [], []
                        t0 = time.time()
                        step = 0
                elif event.key == pygame.K_q:
                    if recording:
                        print("Discarding current episode.")
                        recording = False
                        frames, actions, rewards, timestamps = [], [], [], []
                elif event.key in (pygame.K_UP, pygame.K_w):
                    action = 0
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    action = 1
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    action = 2
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    action = 3

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if recording:
                frames.append(obs.copy())
                actions.append(int(action))
                rewards.append(float(reward))
                timestamps.append(time.time() - t0)
            if terminated or truncated:
                success = bool(terminated)
                if recording and frames:
                    demo_path = env_dir / f"demo_{episode_idx:03d}.hdf5"
                    _save_episode(
                        demo_path,
                        frames=np.stack(frames, axis=0),
                        actions=np.asarray(actions, dtype=np.int64),
                        rewards=np.asarray(rewards, dtype=np.float32),
                        timestamps=np.asarray(timestamps, dtype=np.float64),
                        language=task_instruction,
                        metadata={
                            "env_name": env_name,
                            "player_id": player_id,
                            "date": datetime.utcnow().isoformat(),
                            "success": success,
                        },
                    )
                    print(
                        f"Saved {demo_path.name}: steps={len(frames)}, "
                        f"total_reward={float(np.sum(rewards)):.2f}, success={success}"
                    )
                    episode_idx += 1
                recording = False
                frames, actions, rewards, timestamps = [], [], [], []
                env.reset()

        # Draw recording indicator
        if recording:
            _draw_rec_indicator()

        clock.tick(30)

    pygame.quit()


def replay_demo(demo_path: Path) -> None:
    """Replay a recorded HDF5 demo file."""
    demo_path = demo_path.resolve()
    if not demo_path.exists():
        raise FileNotFoundError(demo_path)

    with h5py.File(demo_path, "r") as f:
        episode = f["episode"]
        observations = episode["observations"][...]  # (T, H, W, 3)

    pygame.init()
    h, w, _ = observations.shape[1:]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"Replay - {demo_path.name}")
    clock = pygame.time.Clock()

    running = True
    t = 0
    T_total = observations.shape[0]
    while running and t < T_total:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        frame = observations[t]
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        t += 1
        clock.tick(30)

    pygame.quit()


def _save_episode(
    path: Path,
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    timestamps: np.ndarray,
    language: str,
    metadata: Dict[str, object],
) -> None:
    """Save a single episode to an HDF5 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode")
        ep.create_dataset("observations", data=frames, compression="gzip", compression_opts=4)
        ep.create_dataset("actions", data=actions)
        ep.create_dataset("rewards", data=rewards)
        ep.create_dataset("timestamps", data=timestamps)
        ep.attrs["language"] = language
        meta_group = ep.create_group("metadata")
        for k, v in metadata.items():
            meta_group.attrs[k] = v


def _sanitize_task(task: str) -> str:
    return (
        task.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace('"', "")
        .replace("'", "")
    )


def _next_episode_index(env_dir: Path) -> int:
    existing = sorted(env_dir.glob("demo_*.hdf5"))
    if not existing:
        return 1
    last = existing[-1].stem
    try:
        return int(last.split("_")[-1]) + 1
    except ValueError:
        return 1


def _draw_rec_indicator() -> None:
    """Draw a red recording dot in the top-right corner of the active window."""
    screen = pygame.display.get_surface()
    if screen is None:
        return
    w, _ = screen.get_size()
    radius = 6
    center = (w - radius - 4, radius + 4)
    pygame.draw.circle(screen, (220, 40, 40), center, radius)

