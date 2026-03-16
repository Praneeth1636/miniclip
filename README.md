# MimicPlay

An imitation learning platform where neural networks learn to play custom 2D games by watching humans. Record gameplay, train visual policies (optionally language-conditioned), and evaluate how well agents generalize.

This repository is under active development. Core components:

- **Custom Pygame environments** with a shared `BaseEnv` interface.
- **HDF5-based recorder** for human demonstrations.
- **Behavioral Cloning (BC)** and **Vision-Language-Action (VLA)** policies in PyTorch.
- **Training and evaluation CLI** with Weights & Biases logging.
- **Streamlit dashboard** for metrics, replays, and GradCAM visualizations.

Full documentation lives in `docs/` (MkDocs with Material theme).
