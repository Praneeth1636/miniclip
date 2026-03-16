# MimicPlay

I wanted to understand how imitation learning actually works — not by reading papers, but by building the full pipeline from scratch. MimicPlay is what came out of that: you play a simple 2D game, the system records your gameplay as visual demonstrations, and a neural network tries to learn your behavior from raw pixels. There's also a language-conditioned mode where you can give the agent instructions like "collect the blue coins" and it learns to follow them.

The architecture mirrors what modern robot learning systems use (RT-1, CLIPort, OpenVLA) — visual observation → policy network → action prediction — just in a much simpler environment where you can actually iterate quickly.

## How It Works

1. You play one of three custom Pygame environments
2. Every frame and keypress gets recorded as an HDF5 demo file
3. A ResNet-18-based policy network trains on your demos via behavioral cloning
4. The trained agent plays the game, and you measure how well it does
5. GradCAM shows you what the model is paying attention to

## Environments

I built three environments with increasing difficulty, all sharing a common `BaseEnv` interface:

| Environment | What It Is | Obs | Actions |
|---|---|---|---|
| **GridCollector** | Top-down grid world, collect coins, avoid walls | 96×96 | 4 |
| **DodgeRunner** | Side-scroller, dodge obstacles, grab power-ups | 128×128 | 5 |
| **BuildBridge** | Puzzle — pick up and place blocks to cross a gap | 128×128 | 5 |

Each environment supports multiple language-conditioned task variants. GridCollector alone has three: "collect all coins", "collect the blue coins", and "reach the green zone."

You can play any of them standalone:

```bash
python -m mimicplay.envs.grid_collector
python -m mimicplay.envs.dodge_runner
python -m mimicplay.envs.build_bridge
```

## Models

**Behavioral Cloning (BC)** — the baseline. Takes a stack of 4 frames, runs them through a ResNet-18 encoder, and predicts the action via an MLP head. Trained with cross-entropy loss and inverse-frequency class weighting (because humans don't press all buttons equally).

**Vision-Language-Action (VLA)** — the interesting one. Same vision encoder, but now there's also a frozen SentenceTransformer that encodes the task instruction. The language embedding modulates the visual features via FiLM conditioning (Feature-wise Linear Modulation) — the same technique used in real robot learning research. This means the model can learn to behave differently depending on what you tell it to do, even in the same environment.

## Quick Start

```bash
git clone https://github.com/Praneeth1636/miniclip.git
cd miniclip
pip install -e ".[dev]"
```

Record some demos (aim for 50+ per task):

```bash
mimicplay record --env grid_collector --task "collect all coins" --player you
```

Controls: `R` to start recording, arrow keys to play, `Q` to discard a bad run, `ESC` to stop.

Train a model:

```bash
mimicplay train --config configs/train_bc.yaml
```

Evaluate:

```bash
mimicplay eval --checkpoint checkpoints/bc_grid_collector_best.pt \
    --env grid_collector --task "collect all coins" --episodes 50
```

There's also a Streamlit dashboard for browsing results, comparing models, and viewing GradCAM attention maps:

```bash
mimicplay dashboard
```

## Project Structure

```
mimicplay/
├── envs/           # Three Pygame environments + shared BaseEnv interface
├── data/           # HDF5 recorder, PyTorch dataset with frame stacking, augmentations
├── models/         # BC policy, VLA policy with FiLM, ResNet encoder
├── training/       # Training loop with class balancing, cosine LR, wandb logging
├── evaluation/     # Evaluator, GradCAM, multi-model comparison
├── dashboard/      # Streamlit app with 5 tabs
└── cli.py          # Record, train, eval, compare, stats, dashboard commands
```

## Design Decisions

**Custom environments, not Gymnasium.** I wanted full control over the rendering, reward structure, and language variant system. Each env is a single file you can read top to bottom.

**HDF5 for demos, not video files.** Random access, compression, and metadata in one file. Standard in robotics research.

**Frame stacking, not recurrence.** 4 frames stacked along the channel dimension gives temporal context without the complexity of LSTMs. The ResNet's first conv layer is modified to accept 12 input channels.

**FiLM conditioning, not concatenation.** For the VLA model, language doesn't just get appended to visual features — it modulates them. This is a meaningful architectural choice that lets the model learn to attend to different visual aspects depending on the instruction.

**No RL.** This is pure imitation learning. The agent never explores on its own or receives reward signals during training. It only learns from watching you play.

## Docs

Detailed documentation in the `docs/` folder:

- [Environments](docs/environments.md) — game design, interfaces, task variants
- [Architecture](docs/architecture.md) — model design, FiLM conditioning, training pipeline
- [Training Guide](docs/training.md) — recording demos, hyperparameters, evaluation
- [Results](docs/results.md) — benchmarks across environments and models

## Tech Stack

Python 3.11+, PyTorch, Pygame, HDF5 (h5py), SentenceTransformers, Weights & Biases, Streamlit, Click/Typer. Linted with ruff, type-checked with mypy.

## Contributing

1. Fork and create a feature branch
2. `pytest tests/` and `ruff check .` before committing
3. Open a PR with a clear description of what and why

## License

MIT
