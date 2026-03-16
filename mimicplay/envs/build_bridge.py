"""BuildBridge environment implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame

from mimicplay.envs.base import BaseEnv


@dataclass
class BridgeConfig:
    width: int = 128
    height: int = 128
    max_steps: int = 300
    blocks_available: int = 6
    gap_start: int = 56
    gap_end: int = 80


class BuildBridgeEnv(BaseEnv):
    """Puzzle environment where the agent builds a bridge and crosses."""

    ACTIONS: List[str] = ["MOVE_LEFT", "MOVE_RIGHT", "PICK_UP", "PLACE", "JUMP"]

    def __init__(self, config: BridgeConfig | None = None) -> None:
        self.config = config or BridgeConfig()
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self._episode: int = 0
        self._step: int = 0
        self._score: float = 0.0

        self._agent_rect: pygame.Rect | None = None
        self._blocks: List[pygame.Rect] = []
        self._carried_block: bool = False
        self._blocks_remaining: int = 0
        self._instruction: str = "build a bridge and cross"

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._ensure_pygame()
        self._episode += 1
        self._step = 0
        self._score = 0.0
        self._blocks = []
        self._carried_block = False
        self._blocks_remaining = self.config.blocks_available

        self._agent_rect = pygame.Rect(10, self.config.height - 24, 10, 14)
        obs = self._render_to_array()
        info: Dict = {
            "episode": self._episode,
            "score": self._score,
            "instruction": self._instruction,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._agent_rect is not None
        self._step += 1
        reward = 0.0

        if action == 0:
            self._agent_rect.x = max(0, self._agent_rect.x - 3)
        elif action == 1:
            self._agent_rect.x = min(self.config.width - self._agent_rect.width, self._agent_rect.x + 3)
        elif action == 2:
            if not self._carried_block and self._blocks_remaining > 0:
                self._carried_block = True
                self._blocks_remaining -= 1
        elif action == 3:
            if self._carried_block:
                block = pygame.Rect(
                    self._agent_rect.x,
                    self.config.height - 16,
                    12,
                    6,
                )
                self._blocks.append(block)
                self._carried_block = False
        elif action == 4:
            # simple jump reward if landing across gap
            if self._agent_rect.x > self.config.gap_end:
                reward += 10.0

        terminated = reward > 0.0
        truncated = self._step >= self.config.max_steps
        self._score += reward

        obs = self._render_to_array()
        info: Dict = {
            "step": self._step,
            "score": self._score,
            "episode": self._episode,
            "terminated": terminated,
            "truncated": truncated,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render_to_array()

    def get_action_space(self) -> List[str]:
        return list(self.ACTIONS)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.config.height, self.config.width, 3)

    def _ensure_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self._screen is None:
            self._screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("MimicPlay - BuildBridge")
        if self._clock is None:
            self._clock = pygame.time.Clock()

    def _render_to_array(self) -> np.ndarray:
        self._ensure_pygame()
        assert self._screen is not None
        assert self._agent_rect is not None

        surface = pygame.Surface((self.config.width, self.config.height))
        surface.fill((40, 28, 18))

        # Ground and gap
        ground_y = self.config.height - 10
        pygame.draw.rect(surface, (180, 140, 90), pygame.Rect(0, ground_y, self.config.width, 10))
        pygame.draw.rect(
            surface,
            (0, 0, 0),
            pygame.Rect(self.config.gap_start, ground_y, self.config.gap_end - self.config.gap_start, 10),
        )

        # Blocks placed
        for block in self._blocks:
            pygame.draw.rect(surface, (210, 180, 120), block)

        # Agent
        pygame.draw.rect(surface, (250, 230, 120), self._agent_rect)

        # Carried block indicator
        if self._carried_block:
            top_block = pygame.Rect(self._agent_rect.x, self._agent_rect.y - 8, 10, 6)
            pygame.draw.rect(surface, (210, 180, 120), top_block)

        self._screen.blit(surface, (0, 0))
        self._draw_ui(self._screen)
        pygame.display.flip()

        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def _draw_ui(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, self.config.width, 14))
        font = pygame.font.SysFont("Arial", 12)
        text = (
            f"Score: {self._score:.1f}  Step: {self._step}  Ep: {self._episode}  "
            f"Blocks left: {self._blocks_remaining}"
        )
        surf = font.render(text, True, (230, 230, 230))
        screen.blit(surf, (4, 0))


def _run_human_play() -> None:
    env = BuildBridgeEnv()
    obs, _ = env.reset()
    _ = obs
    running = True
    clock = pygame.time.Clock()
    while running:
        action = 1  # MOVE_RIGHT by default
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action = 0
        elif keys[pygame.K_RIGHT]:
            action = 1
        elif keys[pygame.K_z]:
            action = 2
        elif keys[pygame.K_x]:
            action = 3
        elif keys[pygame.K_SPACE]:
            action = 4

        obs, reward, terminated, truncated, _ = env.step(action)
        _ = (reward,)
        if terminated or truncated:
            env.reset()
        clock.tick(20)
    pygame.quit()


if __name__ == "__main__":
    _run_human_play()

