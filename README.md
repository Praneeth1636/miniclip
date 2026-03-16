# MimicPlay

An imitation learning platform where neural networks learn to play 2D games by watching humans. Record your gameplay, train a policy from visual demonstrations, and evaluate how well the agent generalizes — including with natural language task instructions.

## How It Works

1. **You play a game.** MimicPlay records every frame and every action as an HDF5 demonstration file.
2. **A neural network trains on your demos.** Either a standard behavioral cloning model (frame → action) or a language-conditioned VLA model (frame + instruction → action).
3. **The agent plays the game.** You evaluate success rate, reward, and use GradCAM to see where the model is looking.

## Environments

| Environment | Observation | Actions | Difficulty |
|-------------|------------|---------|------------|
| GridCollector | 96×96 | 4 (directional) | Easy — good for first experiments |
| DodgeRunner | 128×128 | 5 (directional + stay) | Medium — environment changes without agent input |
| BuildBridge | 128×128 | 5 (move, pick, place, jump) | Hard — requires multi-step planning |

All environments support language-conditioned task variants.

## Models

- **Behavioral Cloning (BC):** ResNet-18 encoder → MLP → action logits. Trained with cross-entropy and inverse-frequency class weighting.
- **Vision-Language-Action (VLA):** Same vision encoder + frozen SentenceTransformer language encoder, fused via FiLM conditioning. A miniature version of the architecture used in modern robot learning (RT-1, CLIPort).

## Quick Start

```bash
git clone https://github.com/Praneeth1636/mimicplay.git
cd mimicplay
pip install -e ".[dev]"

# Play a game
python -m mimicplay.envs.grid_collector

# Record demonstrations
mimicplay record --env grid_collector --task "collect all coins" --player you

# Check your dataset
mimicplay stats demos/grid_collector/

# Train
mimicplay train --config configs/train_bc.yaml

# Evaluate
mimicplay eval --checkpoint checkpoints/bc_grid_collector_best.pt \
    --env grid_collector --task "collect all coins" --episodes 50

# Launch dashboard
mimicplay dashboard
```

## Project Structure

```
mimicplay/
├── envs/           # Pygame environments (GridCollector, DodgeRunner, BuildBridge)
├── data/           # HDF5 recorder, PyTorch dataset, augmentations
├── models/         # BC and VLA policy networks
├── training/       # Training loop, LR scheduling
├── evaluation/     # Evaluator, GradCAM, model comparison
├── dashboard/      # Streamlit app
└── cli.py          # All CLI commands
```

## Documentation

Full docs in the `docs/` folder:
- [Environments](docs/environments.md) — game design, interfaces, language variants
- [Architecture](docs/architecture.md) — model design, FiLM conditioning, training details
- [Training Guide](docs/training.md) — recording demos, configuring training, evaluation
- [Results](docs/results.md) — benchmarks and GradCAM analysis

## Tech Stack

PyTorch, Pygame, HDF5, SentenceTransformers, Weights & Biases, Streamlit, Click/Typer

## Contributing

1. Fork and create a feature branch
2. Run `pytest tests/` and `ruff check .` before committing
3. Open a PR with a clear description

## License

MIT

