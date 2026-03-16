"""Model comparison utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

from mimicplay.evaluation.evaluator import run_evaluation


def run_comparison(
    checkpoint_paths: List[Path],
    env_name: str,
    task_instruction: str,
    num_episodes: int,
) -> None:
    """Compare multiple checkpoints by running evaluation for each."""
    for ckpt in checkpoint_paths:
        print(f"=== Evaluating {ckpt} ===")
        run_evaluation(
            checkpoint_path=ckpt,
            env_name=env_name,
            task_instruction=task_instruction,
            num_episodes=num_episodes,
            record_video=False,
        )

