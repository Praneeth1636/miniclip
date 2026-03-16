"""DodgeRunner environment implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame

from mimicplay.envs.base import BaseEnv


@dataclass
class DodgeConfig:
    width: int = 128
    height: int = 128
    max_steps: int = 500
    obstacle_spawn_prob: float = 0.1
    powerup_spawn_prob: float = 0.02
    base_speed: float = 2.0
    reward_survive: float = 0.1
    reward_collision: float = -10.0
    reward_powerup: float = 5.0


class DodgeRunnerEnv(BaseEnv):
    """Side-scrolling runner with obstacles and power-ups."""

    ACTIONS: List[str] = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    def __init__(self, config: DodgeConfig | None = None) -> None:
        self.config = config or DodgeConfig()
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self._episode: int = 0
        self._step: int = 0
        self._score: float = 0.0

        self._player_rect: pygame.Rect | None = None
        self._obstacles: List[pygame.Rect] = []
        self._powerups: List[pygame.Rect] = []
        self._instruction: str = "avoid all obstacles"

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self._ensure_pygame()
        self._episode += 1
        self._step = 0
        self._score = 0.0
        self._obstacles = []
        self._powerups = []
        self._player_rect = pygame.Rect(20, self.config.height // 2 - 6, 12, 12)
        obs = self._render_to_array()
        info: Dict = {
            "episode": self._episode,
            "score": self._score,
            "instruction": self._instruction,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._player_rect is not None
        self._step += 1
        reward = self.config.reward_survive

        dy = 0
        dx = 0
        if action == 0:
            dy = -3
        elif action == 1:
            dy = 3
        elif action == 2:
            dx = -2
        elif action == 3:
            dx = 2

        self._player_rect.y = max(0, min(self.config.height - self._player_rect.height, self._player_rect.y + dy))
        self._player_rect.x = max(0, min(self.config.width // 2, self._player_rect.x + dx))

        self._spawn_entities()
        self._move_entities()

        terminated = False
        truncated = self._step >= self.config.max_steps

        # Collisions
        for obs_rect in list(self._obstacles):
            if self._player_rect.colliderect(obs_rect):
                reward += self.config.reward_collision
                terminated = True
                break
        for p_rect in list(self._powerups):
            if self._player_rect.colliderect(p_rect):
                reward += self.config.reward_powerup
                self._powerups.remove(p_rect)

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
            pygame.display.setcaption("MimicPlay - DodgeRunner")
        if self._clock is None:
            self._clock = pygame.time.Clock()

    def _spawn_entities(self) -> None:
        if random.random() < self.config.obstacle_spawn_prob:
            h = random.randint(10, 30)
            y = random.randint(0, self.config.height - h)
            rect = pygame.Rect(self.config.width - 10, y, 10, h)
            self._obstacles.append(rect)
        if random.random() < self.config.powerup_spawn_prob:
            y = random.randint(0, self.config.height - 8)
            rect = pygame.Rect(self.config.width - 10, y, 8, 8)
            self._powerups.append(rect)

    def _move_entities(self) -> None:
        vx = -self.config.base_speed
        for rect in list(self._obstacles):
            rect.x += int(vx)
            if rect.right < 0:
                self._obstacles.remove(rect)
        for rect in list(self._powerups):
            rect.x += int(vx)
            if rect.right < 0:
                self._powerups.remove(rect)

    def _render_to_array(self) -> np.ndarray:
        self._ensure_pygame()
        assert self._screen is not None
        assert self._player_rect is not None

        surface = pygame.Surface((self.config.width, self.config.height))
        surface.fill((5, 10, 30))

        # Lane lines
        for y in range(0, self.config.height, 16):
            pygame.draw.line(surface, (20, 40, 80), (0, y), (self.config.width, y), 1)

        # Obstacles
        for rect in self._obstacles:
            pygame.draw.rect(surface, (240, 80, 80), rect, border_radius=2)

        # Powerups
        for rect in self._powerups:
            pygame.draw.rect(surface, (80, 220, 200), rect, border_radius=2)

        # Player
        pygame.draw.rect(surface, (240, 240, 120), self._player_rect, border_radius=3)

        self._screen.blit(surface, (0, 0))
        self._draw_ui(self._screen)
        pygame.display.flip()

        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def _draw_ui(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, self.config.width, 14))
        font = pygame.font.SysFont("Arial", 12)
        text = f"Score: {self._score:.1f}  Step: {self._step}  Ep: {self._episode}"
        surf = font.render(text, True, (220, 220, 220))
        screen.blit(surf, (4, 0))


def _run_human_play() -> None:
    env = DodgeRunnerEnv()
    obs, _ = env.reset()
    _ = obs
    running = True
    clock = pygame.time.Clock()
    while running:
        action = 4  # STAY
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_DOWN]:
            action = 1
        elif keys[pygame.K_LEFT]:
            action = 2
        elif keys[pygame.K_RIGHT]:
            action = 3
        else:
            action = 4

        obs, reward, terminated, truncated, _ = env.step(action)
        _ = (reward,)
        if terminated or truncated:
            env.reset()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    _run_human_play()

