# MimicPlay

MimicPlay is an imitation learning platform where neural networks learn to play 2D games by watching humans. You play a game, your gameplay gets recorded as visual demonstrations, and a policy network learns to replicate your behavior from raw pixel observations.

The platform supports two learning approaches:

- **Behavioral Cloning (BC):** A straightforward supervised learning setup — given a frame, predict the action the human took.
- **Vision-Language-Action (VLA):** A language-conditioned policy that takes both a visual observation and a text instruction (e.g., "collect the blue coins") and predicts the appropriate action. Uses FiLM conditioning to modulate visual features based on language.

## Why This Exists

Most imitation learning research happens in expensive simulation environments with complex robot morphologies. MimicPlay strips the problem down to its core: visual observation → policy → action, trained on a small number of human demonstrations. The environments are simple enough that you can record 50 demos in 20 minutes, train a model in under an hour on a laptop GPU, and see meaningful results.

The architecture mirrors real VLA research (RT-1, CLIPort, OpenVLA) at a fraction of the complexity — making it useful for learning, prototyping, and experimentation.

## Components

- **Three custom Pygame environments** with a shared `BaseEnv` interface
- **HDF5 demonstration recorder** with keyboard controls and episode management
- **BC and VLA policy networks** in PyTorch with ResNet-18 vision encoders
- **Training pipeline** with class-balanced loss, cosine scheduling, and wandb logging
- **Evaluation system** with success rate, reward tracking, and video recording
- **GradCAM visualizations** to see where the model is looking
- **Streamlit dashboard** for metrics, replays, and model comparison
- **Full CLI** for recording, training, evaluation, and analysis

