"""Evaluation routines for trained policies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from mimicplay.envs import make as make_env
from mimicplay.models.bc_policy import BCPolicy
from mimicplay.models.vla_policy import VLAPolicy


def run_evaluation(
    checkpoint_path: Path,
    env_name: str,
    task_instruction: str,
    num_episodes: int,
    record_video: bool,
) -> None:
    """Run evaluation episodes and report basic metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    env = make_env(env_name)
    obs_shape = env.observation_shape
    n_actions = len(env.get_action_space())
    frame_stack = int(cfg.get("frame_stack", 4))
    in_channels = frame_stack * obs_shape[2]

    model_type = cfg.get("model", "bc")
    if model_type == "bc":
        model = BCPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=in_channels)
    else:
        model = VLAPolicy(obs_shape=obs_shape, n_actions=n_actions, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    rewards: List[float] = []
    lengths: List[int] = []
    successes = 0
    action_hist: List[int] = []

    video_dir = Path("eval_videos")
    if record_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames: List[np.ndarray] = []
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            frames.append(obs)
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            with torch.no_grad():
                if isinstance(model, BCPolicy):
                    logits = model(obs_t)
                else:
                    lang_emb = model.encode_language([task_instruction]).to(device)
                    logits = model(obs_t, lang_emb)
                action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_reward += float(r)
            steps += 1

        rewards.append(ep_reward)
        lengths.append(steps)
        if info.get("terminated", False):
            successes += 1
        action_hist.extend(_ for _ in range(steps))

        if record_video:
            _write_video(video_dir / f"ep_{ep:03d}.mp4", frames, fps=30)

    success_rate = 100.0 * successes / max(1, num_episodes)
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_len = float(np.mean(lengths)) if lengths else 0.0

    print(
        f"Eval over {num_episodes} episodes | "
        f"success_rate={success_rate:.1f}% | avg_reward={avg_reward:.2f} | avg_len={avg_len:.1f}"
    )


def _write_video(path: Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

