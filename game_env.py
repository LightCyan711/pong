# -*- coding: utf-8 -*-
"""
Pong — Single-file Human Game + Gymnasium Env
---------------------------------------------
- Window: 800x600, minimalist hyper-casual look
- Human mode (run directly): keyboard controls (W/S or ↑/↓), Space start/restart, ESC pause/quit
- RL mode (import): Gymnasium Env named `GameEnv` with Discrete(3) actions

Action space (agent):
  0 = MOVE_UP, 1 = STAY, 2 = MOVE_DOWN

Observation (8D, all in [-1, 1]):
  [ball_x, ball_y, ball_vx, ball_vy, pad_y, opp_y, rel_y_to_pad, rel_y_to_opp]

Rewards (dense shaping + sparse):
  +1.0  when agent scores
  -1.0  when opponent scores
  +0.01 every step the ball moves toward opponent goal (positive ball_vx)
  +0.05 on paddle hit, scaled by centered contact (closer to paddle center = higher)
  -0.001 * paddle_move_speed (discourages jitter)

Episode ends when either side reaches 11 (match) or time-limit steps elapse (truncation).
"""

import math
import random
from typing import Optional, Tuple, Dict

import numpy as np
import pygame
from pygame import gfxdraw

import gymnasium as gym
from gymnasium import spaces


# ---------------------------- Utilities ---------------------------- #

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------- Environment ---------------------------- #

class GameEnv(gym.Env):
    """
    Gymnasium-compatible Pong environment.
    - Discrete(3) actions: UP / STAY / DOWN
    - Box(8,) observations normalized to [-1, 1]
    - Supports render_mode: None | "human" | "rgb_array"
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Visual constants
    WIDTH, HEIGHT = 800, 600
    PAD_W, PAD_H = 12, 88
    BALL_R = 8
    NET_GAP = 16

    # Physics constants (pixels/second)
    PAD_SPEED = 420.0
    OPP_SPEED = 360.0
    BALL_SPEED_INIT = 340.0
    BALL_SPEED_MAX = 720.0
    BALL_SPEED_GROWTH = 1.025  # on each paddle hit
    SPIN_ANGLE_MAX = math.radians(38)  # max deflection from contact offset

    WIN_SCORE = 11
    STEP_LIMIT = 60 * 60  # 60s at 60 fps

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        self.render_mode = render_mode
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Observation: 8 floats normalized to [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # Action: 0 up, 1 stay, 2 down
        self.action_space = spaces.Discrete(3)

        # Pygame state
        self._screen = None
        self._clock = None
        self._font = None
        self._surface = None  # for rgb_array mode

        # Game state
        self.dt = 1.0 / 60.0
        self.reset_on_point = True  # after each score, re-serve
        self.match_over = False

        self._reset_match()

        if self.render_mode == "human":
            self._init_pygame()

    # -------------------- Core Gym Methods -------------------- #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        # If previous episode ended by match point, start a fresh match
        if self.match_over:
            self._reset_match()
        else:
            self._reset_round(serve_to_right=self.np_random.random() < 0.5)

        self.steps = 0
        obs = self._get_obs()
        info = {"score": (self.score_p, self.score_o)}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"

        # --- Player movement ---
        pad_dir = -1 if action == 0 else (0 if action == 1 else 1)
        old_pad_y = self.pad_y
        self.pad_y += pad_dir * self.PAD_SPEED * self.dt
        self.pad_y = clamp(self.pad_y, self.PAD_H / 2, self.HEIGHT - self.PAD_H / 2)
        pad_move_penalty = 0.001 * abs(self.pad_y - old_pad_y) / (self.PAD_SPEED * self.dt + 1e-6)

        # --- Opponent movement (simple tracking with cap + slight delay) ---
        target = self.ball_y + self.np_random.uniform(-18, 18)
        if self.ball_vx < 0:  # when ball moving back, opp relaxes a bit
            target = self.HEIGHT / 2
        if abs(target - self.opp_y) > 1.0:
            self.opp_y += np.sign(target - self.opp_y) * self.OPP_SPEED * self.dt
        self.opp_y = clamp(self.opp_y, self.PAD_H / 2, self.HEIGHT - self.PAD_H / 2)

        # --- Ball physics ---
        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt

        # Top/Bottom wall bounce
        if self.ball_y <= self.BALL_R:
            self.ball_y = self.BALL_R
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y >= self.HEIGHT - self.BALL_R:
            self.ball_y = self.HEIGHT - self.BALL_R
            self.ball_vy = -abs(self.ball_vy)

        hit_reward = 0.0

        # Paddle rectangles (centered)
        pad_rect = pygame.Rect(0, 0, self.PAD_W, self.PAD_H)
        pad_rect.center = (32, self.pad_y)
        opp_rect = pygame.Rect(0, 0, self.PAD_W, self.PAD_H)
        opp_rect.center = (self.WIDTH - 32, self.opp_y)

        # Collision with player paddle
        if self._circle_rect_collide(self.ball_x, self.ball_y, self.BALL_R, pad_rect) and self.ball_vx < 0:
            self._resolve_paddle_bounce(is_player=True)
            # centered contact bonus: 1 at center, 0 near edge
            offset = (self.ball_y - self.pad_y) / (self.PAD_H / 2)
            centered = 1.0 - min(1.0, abs(offset))
            hit_reward = 0.05 * (0.5 + 0.5 * centered)

        # Collision with opponent paddle
        if self._circle_rect_collide(self.ball_x, self.ball_y, self.BALL_R, opp_rect) and self.ball_vx > 0:
            self._resolve_paddle_bounce(is_player=False)

        # Check scoring
        point_scored = False
        reward = 0.0

        if self.ball_x < -self.BALL_R:  # missed by player -> opponent scores
            self.score_o += 1
            reward = -1.0
            point_scored = True

        elif self.ball_x > self.WIDTH + self.BALL_R:  # opponent misses -> player scores
            self.score_p += 1
            reward = +1.0
            point_scored = True

        # Small shaping: encourage ball moving toward opponent
        reward += 0.01 * (1.0 if self.ball_vx > 0 else -0.005)

        # Penalize unnecessary jittery movement
        reward -= pad_move_penalty

        terminated = False
        truncated = False
        info = {"score": (self.score_p, self.score_o)}

        if point_scored:
            # Match end?
            if self.score_p >= self.WIN_SCORE or self.score_o >= self.WIN_SCORE:
                terminated = True
                self.match_over = True
            else:
                # Reset ball for next rally
                self._reset_round(serve_to_right=(self.score_p + self.score_o) % 2 == 0)

        self.steps += 1
        if self.steps >= self.STEP_LIMIT:
            truncated = True

        # Add any hit reward on top
        reward += hit_reward

        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            # Build an offscreen surface and return ndarray
            if self._surface is None:
                pygame.init()
                self._surface = pygame.Surface((self.WIDTH, self.HEIGHT))
                self._font = pygame.font.SysFont("arial", 28, bold=True)
            self._draw_world(self._surface)
            return pygame.surfarray.array3d(self._surface).swapaxes(0, 1)

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # -------------------- Internal Helpers -------------------- #

    def _reset_match(self):
        self.score_p = 0
        self.score_o = 0
        self.match_over = False
        self.steps = 0
        self._reset_round(serve_to_right=self.np_random.random() < 0.5)

    def _reset_round(self, serve_to_right: bool):
        self.pad_y = self.HEIGHT / 2
        self.opp_y = self.HEIGHT / 2

        self.ball_x = self.WIDTH / 2
        self.ball_y = self.HEIGHT / 2

        angle = self.np_random.uniform(-0.35, 0.35)
        speed = self.BALL_SPEED_INIT
        dir_x = 1.0 if serve_to_right else -1.0

        self.ball_vx = dir_x * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

    def _get_obs(self) -> np.ndarray:
        # Normalize to [-1,1]
        bx = (self.ball_x / self.WIDTH) * 2 - 1
        by = (self.ball_y / self.HEIGHT) * 2 - 1
        # velocity normalized by BALL_SPEED_MAX
        bvx = clamp(self.ball_vx / self.BALL_SPEED_MAX, -1, 1)
        bvy = clamp(self.ball_vy / self.BALL_SPEED_MAX, -1, 1)
        py = (self.pad_y / self.HEIGHT) * 2 - 1
        oy = (self.opp_y / self.HEIGHT) * 2 - 1
        rel_p = clamp((self.ball_y - self.pad_y) / (self.HEIGHT / 2), -1, 1)
        rel_o = clamp((self.ball_y - self.opp_y) / (self.HEIGHT / 2), -1, 1)

        obs = np.array([bx, by, bvx, bvy, py, oy, rel_p, rel_o], dtype=np.float32)
        return obs

    def _resolve_paddle_bounce(self, is_player: bool):
        # Compute contact offset [-1..1] across paddle height
        pad_center_y = self.pad_y if is_player else self.opp_y
        offset = (self.ball_y - pad_center_y) / (self.PAD_H / 2)
        offset = clamp(offset, -1.0, 1.0)

        speed = min(self.BALL_SPEED_MAX, math.hypot(self.ball_vx, self.ball_vy) * self.BALL_SPEED_GROWTH)
        angle = offset * self.SPIN_ANGLE_MAX

        # Player paddle reflects to the right, Opponent to the left
        dir_x = 1.0 if is_player else -1.0
        self.ball_vx = dir_x * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

        # Nudge ball out of paddle to prevent sticking
        self.ball_x += dir_x * (self.PAD_W / 2 + self.BALL_R + 2)

        # Subtle "juicy" feedback: small shake in y
        self.ball_y += self.np_random.uniform(-2.0, 2.0)

    @staticmethod
    def _circle_rect_collide(cx: float, cy: float, cr: float, rect: pygame.Rect) -> bool:
        # Closest point on rect to circle center
        rx = clamp(cx, rect.left, rect.right)
        ry = clamp(cy, rect.top, rect.bottom)
        dx = cx - rx
        dy = cy - ry
        return (dx * dx + dy * dy) <= (cr * cr)

    # -------------------- Rendering -------------------- #

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Pong — Human Mode")
        self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SCALED)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 28, bold=True)

    def _render_frame(self):
        if self._screen is None:
            self._init_pygame()
        self._draw_world(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _draw_world(self, surface: pygame.Surface):
        w, h = self.WIDTH, self.HEIGHT
        # Soft background
        surface.fill((246, 248, 252))  # off-white

        # Center line (dashed)
        cx = w // 2
        dash_h = 12
        for y in range(0, h, dash_h * 2):
            pygame.draw.rect(surface, (220, 228, 236), pygame.Rect(cx - 2, y, 4, dash_h), border_radius=2)

        # Paddles (rounded rectangles)
        pad_color = (64, 112, 244)    # cool blue
        opp_color = (250, 112, 92)    # coral
        self._aa_round_rect(surface, pygame.Rect(26, int(self.pad_y - self.PAD_H / 2), self.PAD_W, self.PAD_H), pad_color, 6)
        self._aa_round_rect(surface, pygame.Rect(w - 26 - self.PAD_W, int(self.opp_y - self.PAD_H / 2), self.PAD_W, self.PAD_H), opp_color, 6)

        # Ball with subtle outline
        bx, by, r = int(self.ball_x), int(self.ball_y), self.BALL_R
        gfxdraw.filled_circle(surface, bx, by, r, (40, 40, 40))
        gfxdraw.aacircle(surface, bx, by, r, (40, 40, 40))
        gfxdraw.filled_circle(surface, bx, by, r - 2, (30, 30, 30))
        gfxdraw.filled_circle(surface, bx, by, r - 3, (255, 186, 73))  # warm yellow core
        gfxdraw.aacircle(surface, bx, by, r - 3, (255, 186, 73))

        # Scores
        score_col = (40, 52, 72)
        s_left = self._font.render(str(self.score_p), True, score_col)
        s_right = self._font.render(str(self.score_o), True, score_col)
        surface.blit(s_left, (w * 0.5 - 60 - s_left.get_width(), 20))
        surface.blit(s_right, (w * 0.5 + 60, 20))

    @staticmethod
    def _aa_round_rect(surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int], radius: int):
        # Draw anti-aliased rounded rectangle
        x, y, w, h = rect
        if radius > min(w, h) / 2:
            radius = int(min(w, h) / 2)
        pygame.draw.rect(surface, color, (x + radius, y, w - 2 * radius, h))
        pygame.draw.rect(surface, color, (x, y + radius, w, h - 2 * radius))
        gfxdraw.filled_circle(surface, x + radius, y + radius, radius, color)
        gfxdraw.filled_circle(surface, x + w - radius - 1, y + radius, radius, color)
        gfxdraw.filled_circle(surface, x + radius, y + h - radius - 1, radius, color)
        gfxdraw.filled_circle(surface, x + w - radius - 1, y + h - radius - 1, radius, color)
        gfxdraw.aacircle(surface, x + radius, y + radius, radius, color)
        gfxdraw.aacircle(surface, x + w - radius - 1, y + radius, radius, color)
        gfxdraw.aacircle(surface, x + radius, y + h - radius - 1, radius, color)
        gfxdraw.aacircle(surface, x + w - radius - 1, y + h - radius - 1, radius, color)


# ---------------------------- Human Runner ---------------------------- #

def run_human():
    """
    Human-playable loop using the Gym env under the hood.
    Controls:
      W / ↑ : move up
      S / ↓ : move down
      SPACE : (re)start serve if point ended / restart match if match over
      ESC   : pause/quit
    """
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    paused = False
    running = True

    while running:
        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # If match over, reset; else do nothing (serve auto happens on step)
                    if env.match_over:
                        obs, info = env.reset()
                    # Otherwise just continue

        keys = pygame.key.get_pressed()
        action = 1  # STAY
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2

        if not paused:
            obs, rew, terminated, truncated, info = env.step(action)
            # If a match has ended, show it and wait for SPACE to reset
            if terminated or truncated:
                # brief overlay
                env._render_frame()
                # Wait until space or quit
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                            running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            obs, info = env.reset()
                            waiting = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                            waiting = False
                            running = False
                    env._render_frame()

    env.close()


# ---------------------------- Main ---------------------------- #

if __name__ == "__main__":
    run_human()
