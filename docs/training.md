# Training Guide

## Prerequisites

1. Record at least 50 demonstrations for your target task (see CLI usage below)
2. Ensure demos are saved under `demos/<env_name>/<task_name>/`

## Recording Demonstrations

```bash
mimicplay record --env grid_collector --task "collect the blue coins" --player your_name
```

Controls during recording:
- **R** — Start recording a new episode
- **Arrow keys** — Play the game
- **Q** — Discard current episode (bad run)
- **ESC** — End session and save

Aim for 50-100 successful demonstrations per task. Quality matters more than quantity.

## Training a BC Model

```bash
mimicplay train --config configs/train_bc.yaml
```

Edit `configs/train_bc.yaml` to point `demo_dir` to your recorded demos. Default hyperparameters work well for GridCollector with 50+ demos.

Key hyperparameters to tune:
- `lr`: Start with 3e-4, reduce if loss is unstable
- `frame_stack`: 4 is default, try 1 to see if temporal context helps
- `batch_size`: 64 works for most cases, reduce if GPU memory is limited
- `epochs`: 100 is usually enough, early stopping prevents overfitting

## Training a VLA Model

```bash
mimicplay train --config configs/train_vla.yaml
```

Same as BC but the model also receives the language instruction. Make sure your demos include the task instruction (set via `--task` during recording).

## Monitoring Training

If wandb is installed, metrics are logged automatically. Otherwise, training loss is printed to stdout each epoch.

## Evaluating

```bash
mimicplay eval --checkpoint checkpoints/bc_grid_collector_best.pt --env grid_collector --task "collect the blue coins" --episodes 50
```

Add `--record` to save evaluation videos as .mp4 files.

