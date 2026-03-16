"""GridCollector environment implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame

from mimicplay.envs.base import BaseEnv


@dataclass
class GridConfig:
    """Configuration parameters for GridCollector."""

    grid_size: int = 12
    cell_size: int = 8
    obs_size: int = 96
    window_scale: int = 4
    max_steps: int = 200
    n_coins: int = 6
    step_penalty: float = -0.01


class GridCollectorEnv(BaseEnv):
    """Top-down grid world where the agent collects coins."""

    ACTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self, config: GridConfig | None = None) -> None:
        self.config = config or GridConfig()
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._episode: int = 0
        self._step: int = 0
        self._score: float = 0.0

        self._agent_pos: Tuple[int, int] = (0, 0)
        self._coins_blue: List[Tuple[int, int]] = []
        self._coins_red: List[Tuple[int, int]] = []
        self._walls: List[Tuple[int, int]] = []

        self._instruction: str = "collect all coins"

    # --- BaseEnv API ---

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._ensure_pygame()
        self._episode += 1
        self._step = 0
        self._score = 0.0

        self._agent_pos = (self.config.grid_size // 2, self.config.grid_size // 2)
        self._walls = self._generate_walls()
        self._coins_blue, self._coins_red = self._generate_coins()

        obs = self._render_to_array()
        info: Dict = {
            "episode": self._episode,
            "score": self._score,
            "instruction": self._instruction,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._step += 1

        dx, dy = self._action_to_delta(action)
        new_pos = (self._agent_pos[0] + dx, self._agent_pos[1] + dy)
        if self._is_free(new_pos):
            self._agent_pos = new_pos

        reward = self.config.step_penalty

        # Coin collection logic – both colors currently count as +1.
        reward += self._collect_coins()
        self._score += reward

        terminated = self._all_coins_collected()
        truncated = self._step >= self.config.max_steps

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
        return (self.config.obs_size, self.config.obs_size, 3)

    # --- Internal helpers ---

    def _ensure_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self._screen is None:
            window_size = self.config.obs_size * self.config.window_scale
            self._screen = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("MimicPlay - GridCollector")
        if self._clock is None:
            self._clock = pygame.time.Clock()

    def _generate_walls(self) -> List[Tuple[int, int]]:
        walls: List[Tuple[int, int]] = []
        g = self.config.grid_size
        # Simple border walls
        for x in range(g):
            walls.append((x, 0))
            walls.append((x, g - 1))
        for y in range(g):
            walls.append((0, y))
            walls.append((g - 1, y))
        return walls

    def _generate_coins(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        blues: List[Tuple[int, int]] = []
        reds: List[Tuple[int, int]] = []
        g = self.config.grid_size
        while len(blues) + len(reds) < self.config.n_coins:
            pos = (random.randint(1, g - 2), random.randint(1, g - 2))
            if pos == self._agent_pos or pos in self._walls or pos in blues or pos in reds:
                continue
            if random.random() < 0.5:
                blues.append(pos)
            else:
                reds.append(pos)
        return blues, reds

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        if action == 0:
            return (0, -1)
        if action == 1:
            return (0, 1)
        if action == 2:
            return (-1, 0)
        if action == 3:
            return (1, 0)
        return (0, 0)

    def _is_free(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or y < 0 or x >= self.config.grid_size or y >= self.config.grid_size:
            return False
        if pos in self._walls:
            return False
        return True

    def _collect_coins(self) -> float:
        reward = 0.0
        if self._agent_pos in self._coins_blue:
            self._coins_blue.remove(self._agent_pos)
            reward += 1.0
        if self._agent_pos in self._coins_red:
            self._coins_red.remove(self._agent_pos)
            reward += 1.0
        return reward

    def _all_coins_collected(self) -> bool:
        return not self._coins_blue and not self._coins_red

    def _render_to_array(self) -> np.ndarray:
        self._ensure_pygame()
        assert self._screen is not None

        g = self.config.grid_size
        cell = self.config.cell_size
        play_size = g * cell

        surface = pygame.Surface((play_size, play_size))
        surface.fill((10, 20, 10))

        # Draw grid
        for x in range(g):
            for y in range(g):
                rect = pygame.Rect(x * cell, y * cell, cell, cell)
                pygame.draw.rect(surface, (30, 60, 30), rect, 1)

        # Draw walls
        for wx, wy in self._walls:
            rect = pygame.Rect(wx * cell, wy * cell, cell, cell)
            pygame.draw.rect(surface, (80, 50, 30), rect)

        # Draw coins
        for cx, cy in self._coins_blue:
            rect = pygame.Rect(cx * cell + 1, cy * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(surface, (70, 160, 255), rect)
        for cx, cy in self._coins_red:
            rect = pygame.Rect(cx * cell + 1, cy * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(surface, (200, 80, 80), rect)

        # Draw agent
        ax, ay = self._agent_pos
        agent_rect = pygame.Rect(ax * cell + 1, ay * cell + 1, cell - 2, cell - 2)
        pygame.draw.rect(surface, (240, 240, 80), agent_rect)
        pygame.draw.rect(surface, (0, 0, 0), agent_rect, 1)

        # Scale to observation size
        surface_scaled = pygame.transform.smoothscale(
            surface, (self.config.obs_size, self.config.obs_size)
        )

        # Upscale for human-visible window while keeping obs_size for the agent
        display_size = self.config.obs_size * self.config.window_scale
        surface_display = pygame.transform.scale(surface_scaled, (display_size, display_size))

        # Blit to screen and draw UI overlay
        self._screen.blit(surface_display, (0, 0))
        self._draw_ui(self._screen)
        pygame.display.flip()

        frame = pygame.surfarray.array3d(surface_scaled)
        frame = np.transpose(frame, (1, 0, 2))  # (H, W, 3)
        return frame

    def _draw_ui(self, screen: pygame.Surface) -> None:
        width, _ = screen.get_size()
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, width, 14))
        font = pygame.font.SysFont("Arial", 12)
        text = f"Score: {self._score:.1f}  Step: {self._step}  Ep: {self._episode}"
        surf = font.render(text, True, (220, 220, 220))
        screen.blit(surf, (4, 0))


def _run_human_play() -> None:
    """Standalone runner for manual play."""
    env = GridCollectorEnv()
    obs, _ = env.reset()
    _ = obs  # unused here but ensures reset is called

    running = True
    clock = pygame.time.Clock()
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

        if action is not None:
            obs, reward, terminated, truncated, _ = env.step(action)
            _ = (obs, reward)  # keep for potential debugging
            if terminated or truncated:
                env.reset()

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    _run_human_play()

