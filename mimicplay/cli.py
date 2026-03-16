"""Command-line interface for MimicPlay."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from mimicplay import __version__

app = typer.Typer(help="MimicPlay CLI: record, train, eval, and visualize imitation learning agents.")


@app.callback()
def main(
    _: bool = typer.Option(
        False,
        "--version",
        help="Show MimicPlay version and exit.",
        callback=lambda value: _print_version_and_exit(value),
        is_eager=True,
    )
) -> None:
    """Root command callback."""


def _print_version_and_exit(show: bool) -> None:
    if show:
        typer.echo(f"MimicPlay {__version__}")
        raise typer.Exit()


@app.command()
def record(
    env: str = typer.Option(..., "--env", "-e", help="Environment name, e.g. grid_collector."),
    task: str = typer.Option(..., "--task", "-t", help="Language instruction for this session."),
    player: str = typer.Option(..., "--player", "-p", help="Player identifier."),
    output_dir: Path = typer.Option(
        Path("demos"),
        "--output-dir",
        "-o",
        help="Base directory to store demonstration files.",
    ),
) -> None:
    """Record human demonstrations in the chosen environment."""
    from mimicplay.data.recorder import run_recording_session

    run_recording_session(env_name=env, task_instruction=task, player_id=player, output_root=output_dir)


@app.command()
def replay(
    demo_path: Path = typer.Argument(..., exists=True, help="Path to a single HDF5 demo file."),
) -> None:
    """Replay a recorded demonstration (no interaction)."""
    from mimicplay.data.recorder import replay_demo

    replay_demo(demo_path)


@app.command()
def stats(
    demo_dir: Path = typer.Argument(..., exists=True, help="Directory containing HDF5 demos."),
) -> None:
    """Compute dataset statistics for a directory of demonstrations."""
    from mimicplay.data.dataset import compute_dataset_stats

    compute_dataset_stats(demo_dir)


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to a YAML config."),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        help="Optional checkpoint path to resume training from.",
    ),
) -> None:
    """Train a policy model from demonstrations."""
    from mimicplay.training.trainer import run_training

    run_training(config_path=config, resume_path=resume)


@app.command()
def eval(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-k", exists=True, help="Model checkpoint."),
    env: str = typer.Option(..., "--env", "-e", help="Environment name."),
    task: str = typer.Option(..., "--task", "-t", help="Language instruction for evaluation episodes."),
    episodes: int = typer.Option(50, "--episodes", "-n", help="Number of evaluation episodes."),
    record_video: bool = typer.Option(
        False,
        "--record",
        help="If set, record evaluation videos to disk.",
    ),
) -> None:
    """Evaluate a trained model in an environment."""
    from mimicplay.evaluation.evaluator import run_evaluation

    run_evaluation(
        checkpoint_path=checkpoint,
        env_name=env,
        task_instruction=task,
        num_episodes=episodes,
        record_video=record_video,
    )


@app.command()
def compare(
    checkpoints: list[Path] = typer.Argument(
        ..., help="One or more checkpoints to compare.", exists=True
    ),
    env: str = typer.Option(..., "--env", "-e", help="Environment name."),
    task: str = typer.Option(..., "--task", "-t", help="Language instruction for evaluation."),
    episodes: int = typer.Option(50, "--episodes", "-n", help="Number of evaluation episodes."),
) -> None:
    """Compare multiple trained models on the same task."""
    from mimicplay.evaluation.compare import run_comparison

    run_comparison(
        checkpoint_paths=checkpoints,
        env_name=env,
        task_instruction=task,
        num_episodes=episodes,
    )


@app.command()
def dashboard() -> None:
    """Launch the Streamlit evaluation dashboard."""
    from mimicplay.dashboard.app import launch_dashboard

    launch_dashboard()


if __name__ == "__main__":
    app()

