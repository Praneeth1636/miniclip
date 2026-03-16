# Environments

MimicPlay includes three custom environments built with Pygame. Each follows the `BaseEnv` interface defined in `mimicplay/envs/base.py`.

## Shared Interface

All environments implement:
- `reset()` → observation, info dict
- `step(action)` → observation, reward, terminated, truncated, info
- `render()` → RGB numpy array
- `get_action_space()` → list of action names
- `observation_shape` → (H, W, C) tuple

## GridCollector

A top-down 12x12 grid world. The agent (yellow square) navigates around walls to collect coins. Blue and red coins spawn randomly. Episode ends when all coins are collected or after 200 steps.

- **Observation:** 96×96 RGB
- **Actions:** UP, DOWN, LEFT, RIGHT
- **Reward:** +1.0 per coin collected, -0.01 per step
- **Language variants:** "collect all coins", "collect the blue coins", "reach the green zone"

This is the simplest environment and the best starting point for training and debugging.

## DodgeRunner

A side-scrolling survival game. Obstacles scroll from right to left at varying speeds. The agent must dodge obstacles and optionally collect power-ups.

- **Observation:** 128×128 RGB
- **Actions:** UP, DOWN, LEFT, RIGHT, STAY
- **Reward:** +0.1 per frame survived, -10.0 on collision, +5.0 per power-up
- **Language variants:** "avoid all obstacles", "collect power-ups while staying alive", "stay in the top half"

More challenging than GridCollector because the environment changes even when the agent does nothing.

## BuildBridge

A puzzle environment requiring multi-step planning. The agent must pick up blocks and place them to build a bridge across a gap.

- **Observation:** 128×128 RGB
- **Actions:** MOVE_LEFT, MOVE_RIGHT, PICK_UP, PLACE, JUMP
- **Reward:** Sparse — +10.0 for reaching the other side
- **Language variants:** "build a bridge and cross", "stack three blocks", "cross using the fewest blocks"

The hardest environment. Sparse rewards mean behavioral cloning needs high-quality demonstrations to learn anything useful.

## Running Environments Standalone

Each environment can be played directly:

```bash
python -m mimicplay.envs.grid_collector
python -m mimicplay.envs.dodge_runner
python -m mimicplay.envs.build_bridge
```

Arrow keys to move, ESC to quit.

