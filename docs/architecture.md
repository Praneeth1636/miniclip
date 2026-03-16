# Architecture

## Vision Encoder

Both BC and VLA policies use a ResNet-18 backbone pretrained on ImageNet. The first convolutional layer is replaced to accept `frame_stack * 3` input channels (default: 4 frames × 3 channels = 12 channels). The final fully-connected layer is replaced with an identity function, producing a 512-dimensional feature vector.

## Frame Stacking

Single frames lose motion information. The dataset stacks the last 4 frames along the channel dimension, giving the model temporal context. For timestep `t`, the input is frames `[t-3, t-2, t-1, t]` concatenated along channels.

## Behavioral Cloning (BC)

```
Stacked Frames (B, 12, H, W)
    → ResNet-18 encoder → (B, 512)
    → Linear(512, 256) → ReLU → Dropout(0.3)
    → Linear(256, 128) → ReLU → Dropout(0.2)
    → Linear(128, n_actions)
    → Cross-Entropy Loss
```

Trained as standard supervised classification: given a frame stack, predict the human's action.

## Vision-Language-Action (VLA)

```
Stacked Frames (B, 12, H, W)             Language Instruction
    → ResNet-18 encoder → (B, 512)          → SentenceTransformer → (B, 384)
                                             → Linear(384, 512)
                                                   ↓
                                         FiLM Conditioning:
                                           γ = Linear(512, 512)
                                           β = Linear(512, 512)
                                           fused = γ * vis_features + β
                                                   ↓
                                         → MLP head → action logits
```

FiLM (Feature-wise Linear Modulation) lets the language instruction control *how* visual features are processed. Different instructions cause the model to attend to different visual aspects — e.g., "collect blue coins" should make the model focus on blue objects.

The language encoder (all-MiniLM-L6-v2) is frozen during training. Only the projection layer and FiLM parameters are learned.

## Training

- **Loss:** Cross-entropy with inverse-frequency class weights (humans press some buttons more than others)
- **Optimizer:** AdamW with cosine annealing
- **Data augmentation:** Random crop (pad 8px, crop back), brightness/contrast jitter
- **Checkpointing:** Best model saved based on training loss
- **Logging:** Weights & Biases (optional, graceful fallback if not installed)

## Data Format

Demonstrations are stored as HDF5 files with gzip-compressed observation arrays. Each file contains one episode with observations, actions, rewards, timestamps, language instruction, and metadata (env name, player ID, date, success flag).

