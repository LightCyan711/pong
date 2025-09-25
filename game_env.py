# -*- coding: utf-8 -*-
"""
HyperPong — Local Modes (Polar / TimeRally / Gateball / Beat) + Gymnasium Env
-----------------------------------------------------------------------------

로컬 싱글파일:
- Window: 800x600
- Human mode: W/S 또는 ↑/↓, ESC pause/quit, SPACE 재시작
- 시작 시 텍스트 UI에서 모드 선택 (1~5)
- Gymnasium Env: GameEnv(action=Discrete(3), obs=8D) — 기존 호환 유지

모드 목록 (LOCAL ONLY):
1) Classic (기본 퐁)
2) Polar Pong (극성 스위칭: E)
3) Time Rally (타임슬립/에코)
4) Gateball (포탈/중력우물)
5) Beat Pong (BPM 타이밍 패링)

Author: you
"""

import math
import random
from typing import Optional, Tuple, Dict, Deque, List
from collections import deque

import numpy as np
import pygame
from pygame import gfxdraw

import gymnasium as gym
from gymnasium import spaces


# ---------------------------- Utilities ---------------------------- #

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ---------------------------- Environment ---------------------------- #

class GameEnv(gym.Env):
    """
    Gymnasium-compatible Pong environment (with local-only modes).
    - Discrete(3) actions: 0=UP / 1=STAY / 2=DOWN
    - Box(8,) observations normalized to [-1, 1]
    - render_mode: None | "human" | "rgb_array"
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Visual constants
    WIDTH, HEIGHT = 800, 600
    PAD_W, PAD_H = 12, 88
    BALL_R = 8
    NET_GAP = 16

    # Physics constants (pixels/second) — 약간 빡세게
    PAD_SPEED = 480.0
    OPP_SPEED = 380.0
    BALL_SPEED_INIT = 420.0
    BALL_SPEED_MAX = 1300.0
    BALL_GROW_HIT = 1.14     # 패들 히트 가속
    BALL_GROW_WALL = 1.03    # 벽 반사 미세 가속
    SPIN_ANGLE_MAX = math.radians(40)  # 최대 반사편향각

    WIN_SCORE = 11
    STEP_LIMIT = 60 * 90  # 90s at 60 fps

    # Local Modes
    MODE_CLASSIC   = "classic"
    MODE_POLAR     = "polar"
    MODE_TIMERALLY = "timerally"
    MODE_GATEBALL  = "gateball"
    MODE_BEAT      = "beat"

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None,
                 local_mode: str = MODE_CLASSIC):
        super().__init__()
        self.render_mode = render_mode
        self.local_mode = local_mode
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
        self.reset_on_point = True
        self.match_over = False

        # VFX helpers
        self.shake = 0.0
        self.sparks: List[dict] = []
        self.trail: Deque[tuple] = deque(maxlen=110)
        self.score_pop = None

        # Mode-specific states
        self._init_mode_states()

        self._reset_match()

        if self.render_mode == "human":
            self._init_pygame()

    # -------------------- Mode init -------------------- #

    def _init_mode_states(self):
        # Polar: paddle polarity + UI tint
        self.polar_player = 1   # +1=Blue, -1=Red
        self.polar_opp    = -1
        self.polar_tint   = 0.0  # UI frame swell

        # Time Rally: history for rewind
        self.hist: Deque[tuple] = deque(maxlen=int(0.8 / self.dt))  # ~0.8s
        self.timerally_prob = 0.12  # 히트 시 소확률 타임슬립
        self.echo_age = 0.0
        self.echo_enabled = True

        # Gateball: portals + gravity wells
        self.portal_a = None  # (x,y,r)
        self.portal_b = None
        self.portal_t = 0.0
        self.wells = []       # [(x,y,strength), ...]
        self._spawn_gateball_features()

        # Beat Pong: metronome/BPM & combo
        self.beat_bpm = 120
        self.beat_period = 60.0 / self.beat_bpm
        self.beat_time = 0.0
        self.beat_window = 0.150  # ±150ms
        self.beat_combo = 0
        self.beat_last_good = False

    def _spawn_gateball_features(self):
        if self.local_mode != self.MODE_GATEBALL:
            self.portal_a = self.portal_b = None
            self.wells = []
            return
        w, h = self.WIDTH, self.HEIGHT
        # 랜덤 포탈 한 쌍
        self.portal_a = (self.np_random.uniform(w*0.25, w*0.4),
                         self.np_random.uniform(h*0.25, h*0.75),
                         22.0)
        self.portal_b = (self.np_random.uniform(w*0.6, w*0.75),
                         self.np_random.uniform(h*0.25, h*0.75),
                         22.0)
        # 약한 중력장 1~2개
        self.wells = []
        if self.np_random.random() < 0.7:
            self.wells.append((w*0.5, h*0.35, 12000.0))
        if self.np_random.random() < 0.4:
            self.wells.append((w*0.5, h*0.65, 9000.0))
        self.portal_t = 0.0

    # -------------------- Core Gym Methods -------------------- #

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        if self.match_over:
            self._reset_match()
        else:
            self._reset_round(serve_to_right=self.np_random.random() < 0.5)

        self.steps = 0
        self.hist.clear()
        self.trail.clear()
        self.sparks.clear()
        self.score_pop = None
        self.polar_tint = 0.0
        self.beat_time = 0.0
        self.beat_combo = 0
        self.beat_last_good = False
        self._spawn_gateball_features()

        obs = self._get_obs()
        info = {"score": (self.score_p, self.score_o)}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        self.beat_time += self.dt
        self.portal_t += self.dt

        # --- Save history (Time Rally) ---
        if self.local_mode == self.MODE_TIMERALLY:
            self.hist.append((self.ball_x, self.ball_y, self.ball_vx, self.ball_vy))

        # --- Player movement ---
        pad_dir = -1 if action == 0 else (0 if action == 1 else 1)
        old_pad_y = self.pad_y
        self.pad_y += pad_dir * self.PAD_SPEED * self.dt
        self.pad_y = clamp(self.pad_y, self.PAD_H / 2, self.HEIGHT - self.PAD_H / 2)
        pad_move_penalty = 0.001 * abs(self.pad_y - old_pad_y) / (self.PAD_SPEED * self.dt + 1e-6)

        # --- Opponent simple policy + (Polar) polarity mind-game ---
        target = self.ball_y + self.np_random.uniform(-16, 16)
        if self.ball_vx < 0:
            target = lerp(target, self.HEIGHT/2, 0.6)
        if abs(target - self.opp_y) > 1.0:
            self.opp_y += np.sign(target - self.opp_y) * self.OPP_SPEED * self.dt
        self.opp_y = clamp(self.opp_y, self.PAD_H / 2, self.HEIGHT - self.PAD_H / 2)

        if self.local_mode == self.MODE_POLAR:
            # 간단한 AI - 공의 극성/접근각에 따라 가끔 극성 스위치
            if self.np_random.random() < 0.02:
                prefer_same = (self.ball_vx > 0)  # 자기에게 올 때 반대극성(감속)도 선택적으로
                if prefer_same and self.polar_opp != self.ball_polar():
                    self.polar_opp *= -1
                elif not prefer_same and self.polar_opp == self.ball_polar():
                    self.polar_opp *= -1

        # --- Ball physics integration ---
        # (Gateball) gravity wells
        if self.local_mode == self.MODE_GATEBALL and self.wells:
            ax, ay = 0.0, 0.0
            for (wx, wy, g) in self.wells:
                dx = wx - self.ball_x; dy = wy - self.ball_y
                d2 = dx*dx + dy*dy + 1e-3
                inv = g / d2
                ax += dx * inv * self.dt
                ay += dy * inv * self.dt
            self.ball_vx += ax * 0.03
            self.ball_vy += ay * 0.03

        self.ball_x += self.ball_vx * self.dt
        self.ball_y += self.ball_vy * self.dt

        # Top/Bottom wall bounce (+growth)
        if self.ball_y <= self.BALL_R:
            self.ball_y = self.BALL_R
            self.ball_vy = abs(self.ball_vy)
            self._grow_wall()
        elif self.ball_y >= self.HEIGHT - self.BALL_R:
            self.ball_y = self.HEIGHT - self.BALL_R
            self.ball_vy = -abs(self.ball_vy)
            self._grow_wall()

        # Gateball — Portals warp
        if self.local_mode == self.MODE_GATEBALL and self.portal_a and self.portal_b:
            self._try_portal_warp()

        hit_reward = 0.0

        # Paddles rects
        pad_rect = pygame.Rect(0, 0, self.PAD_W, self.PAD_H)
        pad_rect.center = (32, self.pad_y)
        opp_rect = pygame.Rect(0, 0, self.PAD_W, self.PAD_H)
        opp_rect.center = (self.WIDTH - 32, self.opp_y)

        # Collision: Player
        if self._circle_rect_collide(self.ball_x, self.ball_y, self.BALL_R, pad_rect) and self.ball_vx < 0:
            self._resolve_paddle_bounce(is_player=True)
            # centered bonus
            offset = (self.ball_y - self.pad_y) / (self.PAD_H / 2)
            centered = 1.0 - min(1.0, abs(offset))
            hit_reward = 0.05 * (0.5 + 0.5 * centered)

            # Time Rally chance
            if self.local_mode == self.MODE_TIMERALLY and self.hist and self.np_random.random() < self.timerally_prob:
                self._perform_timeslip()

            # Beat Pong timing window
            if self.local_mode == self.MODE_BEAT:
                in_window = self._is_in_beat_window()
                self.beat_last_good = in_window
                if in_window:
                    self.beat_combo += 1
                    self._boost_beat()
                else:
                    self.beat_combo = 0

        # Collision: Opponent
        if self._circle_rect_collide(self.ball_x, self.ball_y, self.BALL_R, opp_rect) and self.ball_vx > 0:
            self._resolve_paddle_bounce(is_player=False)
            if self.local_mode == self.MODE_TIMERALLY and self.hist and self.np_random.random() < self.timerally_prob*0.6:
                self._perform_timeslip()

        # Scoring
        point_scored = False
        reward = 0.0
        if self.ball_x < -self.BALL_R:
            self.score_o += 1; reward = -1.0; point_scored = True
        elif self.ball_x > self.WIDTH + self.BALL_R:
            self.score_p += 1; reward = +1.0; point_scored = True

        # Shaping
        reward += 0.01 * (1.0 if self.ball_vx > 0 else -0.005)
        reward -= pad_move_penalty
        reward += hit_reward

        terminated = False
        truncated = False
        info = {"score": (self.score_p, self.score_o)}

        if point_scored:
            if self.score_p >= self.WIN_SCORE or self.score_o >= self.WIN_SCORE:
                terminated = True
                self.match_over = True
            else:
                self._reset_round(serve_to_right=(self.score_p + self.score_o) % 2 == 0)

        self.steps += 1
        if self.steps >= self.STEP_LIMIT:
            truncated = True

        # VFX updates
        self._update_vfx()

        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()

        return obs, float(reward), terminated, truncated, info

    # -------------------- Helpers (physics & modes) -------------------- #

    def _grow_wall(self):
        sp = min(self.BALL_SPEED_MAX, math.hypot(self.ball_vx, self.ball_vy) * self.BALL_GROW_WALL)
        ang = math.atan2(self.ball_vy, self.ball_vx)
        self.ball_vx = math.cos(ang) * sp
        self.ball_vy = math.sin(ang) * sp
        self._shake(4)

    def _resolve_paddle_bounce(self, is_player: bool):
        pad_center_y = self.pad_y if is_player else self.opp_y
        offset = clamp((self.ball_y - pad_center_y) / (self.PAD_H / 2), -1.0, 1.0)

        # Base angle
        angle = offset * self.SPIN_ANGLE_MAX
        sp = min(self.BALL_SPEED_MAX, math.hypot(self.ball_vx, self.ball_vy) * self.BALL_GROW_HIT)
        dir_x = 1.0 if is_player else -1.0

        # Polar mode modifications
        if self.local_mode == self.MODE_POLAR:
            pp = self.polar_player if is_player else self.polar_opp
            bp = self.ball_polar()  # 공의 극성: vx>0 -> +1(Blue), vx<0 -> -1(Red) 로 단순 정의
            if pp == bp:
                # 같은 극성: 가속 + 각도 안정(편향 감소)
                sp *= 1.10
                angle *= 0.7
                self.polar_tint = min(1.0, self.polar_tint + 0.6)
            else:
                # 반대 극성: 감속 + 난각(조금 랜덤)
                sp *= 0.90
                angle *= 1.15
                angle += self.np_random.uniform(-0.08, 0.08)
                self.polar_tint = min(1.0, self.polar_tint + 0.35)

        self.ball_vx = dir_x * sp * math.cos(angle)
        self.ball_vy = sp * math.sin(angle)
        self.ball_x += dir_x * (self.PAD_W / 2 + self.BALL_R + 2)
        self.ball_y += self.np_random.uniform(-2.0, 2.0)

        # 스파크
        self._spawn_hit_sparks(self.ball_x - dir_x*8, self.ball_y, dir_x)
        self._shake(8 * min(1.0, sp / self.BALL_SPEED_MAX))

    def _perform_timeslip(self):
        # 과거 0.3~0.6s 프레임으로 순간이동
        frames = int(self.np_random.uniform(0.3, 0.6) / self.dt)
        if frames < 2 or len(self.hist) < frames:
            return
        idx = -frames
        hx, hy, hvx, hvy = self.hist[idx]
        self.ball_x, self.ball_y = hx, hy
        # 속도는 유지하되 약간 글리치 느낌
        self.ball_vx *= 0.98
        self.ball_vy *= 0.98
        self._shake(10)
        self.echo_age = 0.3  # 렌더링에서 글리치/에코 효과

    def _try_portal_warp(self):
        (ax, ay, ar) = self.portal_a
        (bx, by, br) = self.portal_b
        if (self.ball_x - ax)**2 + (self.ball_y - ay)**2 <= (ar + self.BALL_R)**2:
            # A -> B
            ang = math.atan2(self.ball_vy, self.ball_vx) + self.np_random.uniform(-0.15, 0.15)
            sp  = math.hypot(self.ball_vx, self.ball_vy)
            self.ball_x, self.ball_y = bx, by
            self.ball_vx, self.ball_vy = math.cos(ang)*sp, math.sin(ang)*sp
            self._shake(9)
        elif (self.ball_x - bx)**2 + (self.ball_y - by)**2 <= (br + self.BALL_R)**2:
            # B -> A
            ang = math.atan2(self.ball_vy, self.ball_vx) + self.np_random.uniform(-0.15, 0.15)
            sp  = math.hypot(self.ball_vx, self.ball_vy)
            self.ball_x, self.ball_y = ax, ay
            self.ball_vx, self.ball_vy = math.cos(ang)*sp, math.sin(ang)*sp
            self._shake(9)

        # 8~12초마다 배치 리롤
        if self.portal_t > self.np_random.uniform(8.0, 12.0):
            self._spawn_gateball_features()

    def _boost_beat(self):
        # 비트 온-타이밍 성공 시 강화 반사
        sp = math.hypot(self.ball_vx, self.ball_vy) * (1.08 + min(0.12, self.beat_combo*0.02))
        ang = math.atan2(self.ball_vy, self.ball_vx)
        sp = min(self.BALL_SPEED_MAX, sp)
        self.ball_vx = math.cos(ang) * sp
        self.ball_vy = math.sin(ang) * sp
        self._shake(10)

    def _is_in_beat_window(self) -> bool:
        # beat_time을 period로 모듈러 → 중심(0)에 가까울수록 타이밍 정확
        t = self.beat_time % self.beat_period
        dist = min(t, self.beat_period - t)
        return dist <= self.beat_window

    def ball_polar(self) -> int:
        # 단순 정의: vx >= 0 => Blue(+1), vx < 0 => Red(-1)
        return 1 if self.ball_vx >= 0 else -1

    # -------------------- Internal Helpers -------------------- #

    def _reset_match(self):
        self.score_p = 0
        self.score_o = 0
        self.match_over = False
        self.steps = 0
        # 시작 시 공 극성은 vx로 자연 결정
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

        # 라운드 시작 시 약간의 트레일 정리
        self.trail.clear()
        self.sparks.clear()
        self.echo_age = 0.0

    def _get_obs(self) -> np.ndarray:
        bx = (self.ball_x / self.WIDTH) * 2 - 1
        by = (self.ball_y / self.HEIGHT) * 2 - 1
        bvx = clamp(self.ball_vx / self.BALL_SPEED_MAX, -1, 1)
        bvy = clamp(self.ball_vy / self.BALL_SPEED_MAX, -1, 1)
        py = (self.pad_y / self.HEIGHT) * 2 - 1
        oy = (self.opp_y / self.HEIGHT) * 2 - 1
        rel_p = clamp((self.ball_y - self.pad_y) / (self.HEIGHT / 2), -1, 1)
        rel_o = clamp((self.ball_y - self.opp_y) / (self.HEIGHT / 2), -1, 1)
        obs = np.array([bx, by, bvx, bvy, py, oy, rel_p, rel_o], dtype=np.float32)
        return obs

    @staticmethod
    def _circle_rect_collide(cx: float, cy: float, cr: float, rect: pygame.Rect) -> bool:
        rx = clamp(cx, rect.left, rect.right)
        ry = clamp(cy, rect.top, rect.bottom)
        dx = cx - rx
        dy = cy - ry
        return (dx * dx + dy * dy) <= (cr * cr)

    def _spawn_hit_sparks(self, x: float, y: float, dir_sign: float):
        for _ in range(10):
            ang = (self.np_random.random()*0.9 - 0.45) + (0 if dir_sign>0 else math.pi)
            spd = 220 + self.np_random.random()*420
            self.sparks.append({"x":x,"y":y,"vx":math.cos(ang)*spd,"vy":math.sin(ang)*spd,"life":0.25,"age":0.0})

    def _update_vfx(self):
        # sparks
        for i in range(len(self.sparks)-1, -1, -1):
            s = self.sparks[i]; s["age"] += self.dt
            s["x"] += s["vx"] * self.dt; s["y"] += s["vy"] * self.dt
            s["vx"] *= 0.96; s["vy"] *= 0.96
            if s["age"] >= s["life"]:
                self.sparks.pop(i)

        # trail (flame)
        sp = math.hypot(self.ball_vx, self.ball_vy)
        heat = clamp((sp - self.BALL_SPEED_INIT) / (self.BALL_SPEED_MAX - self.BALL_SPEED_INIT), 0, 1)
        self.trail.appendleft((self.ball_x, self.ball_y, heat))

        # echo age decay (Time Rally)
        if self.echo_age > 0.0:
            self.echo_age = max(0.0, self.echo_age - self.dt)

        # polar tint decay
        if self.polar_tint > 0.0:
            self.polar_tint = max(0.0, self.polar_tint - self.dt * 1.2)

        # shake decay
        if self.shake > 0.0:
            self.shake *= 0.85
            if self.shake < 0.2:
                self.shake = 0.0

    def _shake(self, power: float):
        self.shake = max(self.shake, power)

    # -------------------- Rendering -------------------- #

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("HyperPong — Local Modes")
        self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SCALED)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("arial", 28, bold=True)
        self._font_small = pygame.font.SysFont("arial", 16, bold=True)

    def _render_frame(self):
        if self._screen is None:
            self._init_pygame()
        self._draw_world(self._screen)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def _draw_world(self, surface: pygame.Surface):
        w, h = self.WIDTH, self.HEIGHT
        # Background (neon-ish)
        g0 = pygame.Surface((w, h))
        g0.fill((13, 20, 52))
        g1 = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(g1, (15, 22, 64, 180), g1.get_rect(), border_radius=0)
        surface.blit(g0, (0,0))
        surface.blit(g1, (0,0))

        # Polar tint frame
        if self.local_mode == self.MODE_POLAR and self.polar_tint > 0.0:
            c = (90, 150, 255) if self.ball_polar() > 0 else (255, 110, 90)
            alpha = int(120 * self.polar_tint)
            frame = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(frame, c + (alpha,), frame.get_rect(), width=10, border_radius=8)
            surface.blit(frame, (0,0))

        # Center dashed line (Classic / Polar / TimeRally / Beat)
        if self.local_mode in (self.MODE_CLASSIC, self.MODE_POLAR, self.MODE_TIMERALLY, self.MODE_BEAT):
            cx = w // 2
            for y in range(0, h, 24):
                pygame.draw.rect(surface, (38, 50, 100), pygame.Rect(cx - 2, y, 4, 12), border_radius=2)

        # Gateball décor
        if self.local_mode == self.MODE_GATEBALL:
            # Arena grid
            for x in range(0, w, 28):
                pygame.draw.rect(surface, (30, 44, 88), pygame.Rect(x, h//2-1, 14, 2))
            for y in range(0, h, 28):
                pygame.draw.rect(surface, (30, 44, 88), pygame.Rect(w//2-1, y, 2, 14))
            # Portals
            if self.portal_a and self.portal_b:
                for (px, py, pr), col in [(self.portal_a, (120,200,255)), (self.portal_b, (255,160,120))]:
                    pygame.draw.circle(surface, col, (int(px), int(py)), int(pr), width=3)
            # Wells
            for (wx, wy, _) in self.wells:
                pygame.draw.circle(surface, (110,140,255), (int(wx), int(wy)), 6)

        # Echo/Glitch overlay (Time Rally)
        if self.local_mode == self.MODE_TIMERALLY and self.echo_age > 0.0:
            a = int(120 * min(1.0, self.echo_age / 0.3))
            glitch = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(glitch, (180, 220, 255, a), glitch.get_rect(), width=3)
            surface.blit(glitch, (0,0))

        # Screen shake
        ox = oy = 0
        if self.shake > 0.0:
            ox = int((self.np_random.random()*2 - 1) * self.shake)
            oy = int((self.np_random.random()*2 - 1) * self.shake)

        # Trails (flame)
        self._draw_flame_trail(surface, ox, oy)

        # Paddles
        pad_color = (91, 124, 255)   # L
        opp_color = (255, 110, 90)   # R
        self._aa_round_rect(surface, pygame.Rect(26+ox, int(self.pad_y - self.PAD_H / 2)+oy, self.PAD_W, self.PAD_H), pad_color, 6)
        self._aa_round_rect(surface, pygame.Rect(w - 26 - self.PAD_W + ox, int(self.opp_y - self.PAD_H / 2)+oy, self.PAD_W, self.PAD_H), opp_color, 6)

        # Ball
        bx, by, r = int(self.ball_x+ox), int(self.ball_y+oy), self.BALL_R
        gfxdraw.filled_circle(surface, bx, by, r, (22, 22, 30))
        gfxdraw.aacircle(surface, bx, by, r, (22, 22, 30))
        gfxdraw.filled_circle(surface, bx, by, r - 2, (28, 32, 44))
        gfxdraw.filled_circle(surface, bx, by, r - 3, (255, 200, 90))
        gfxdraw.aacircle(surface, bx, by, r - 3, (255, 200, 90))

        # Sparks
        for s in self.sparks:
            t = 1.0 - s["age"]/s["life"]
            col = (255, 180, 80, int(150*t))
            pygame.gfxdraw.filled_circle(surface, int(s["x"]), int(s["y"]), int(2+3*t), (col[0], col[1], col[2]))

        # Scores
        score_col = (210, 224, 255)
        s_left = self._font.render(str(self.score_p), True, score_col)
        s_right = self._font.render(str(self.score_o), True, score_col)
        surface.blit(s_left, (w * 0.5 - 60 - s_left.get_width(), 20))
        surface.blit(s_right, (w * 0.5 + 60, 20))

        # Beat combo HUD
        if self.local_mode == self.MODE_BEAT and self.beat_combo > 0:
            txt = f"COMBO x{self.beat_combo}"
            hud = self._font_small.render(txt, True, (255, 240, 180))
            surface.blit(hud, (w//2 - hud.get_width()//2, 56))

    def _draw_flame_trail(self, surface: pygame.Surface, ox: int, oy: int):
        # speed-reactive flame trail + additive glow
        if not self.trail:
            return
        sp_now = math.hypot(self.ball_vx, self.ball_vy)
        heat_now = clamp((sp_now - self.BALL_SPEED_INIT)/(self.BALL_SPEED_MAX - self.BALL_SPEED_INIT), 0, 1)
        length = int(6 + heat_now*32)

        # glow pass
        for i, (x, y, heat) in enumerate(list(self.trail)[:length]):
            t = i / max(1, length-1)
            a = (1 - t) * (0.13 + 0.6*heat)
            r = int(self.BALL_R * (1 + 0.95*(1-t)*(0.3+0.7*heat)))
            col_outer = (255, 160, 60, int(120*a))
            col_core  = (255, 220,100, int(200*a))
            pygame.gfxdraw.filled_circle(surface, int(x+ox), int(y+oy), int(r*1.8), (col_outer[0], col_outer[1], col_outer[2]))
            pygame.gfxdraw.filled_circle(surface, int(x+ox), int(y+oy), max(2, int(r*0.6)), (col_core[0], col_core[1], col_core[2]))

    @staticmethod
    def _aa_round_rect(surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int], radius: int):
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


# ---------------------------- Human Runner (LOCAL MODES) ---------------------------- #

def run_human():
    """
    Human-playable loop using the Gym env with local-mode selection.
    Controls:
      W / ↑ : move up
      S / ↓ : move down
      E     : (Polar 모드일 때) 극성 토글
      SPACE : 라운드 재시작(매치 종료 후)
      ESC   : 종료
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("arial", 36, bold=True)
    font = pygame.font.SysFont("arial", 22, bold=True)

    # Title: local-mode select
    selecting = True
    selected_mode = GameEnv.MODE_CLASSIC

    def draw_title():
        screen.fill((10, 14, 32))
        t0 = font_big.render("HyperPong — Local Modes", True, (230, 240, 255))
        screen.blit(t0, (400 - t0.get_width()//2, 80))
        lines = [
            "1) Classic",
            "2) Polar Pong   (E: toggle polarity)",
            "3) Time Rally   (random timeslip)",
            "4) Gateball     (portals + gravity wells)",
            "5) Beat Pong    (BPM timing parry)"
        ]
        for i, s in enumerate(lines):
            surf = font.render(s, True, (190, 205, 255))
            screen.blit(surf, (240, 160 + i*36))
        inst = font.render("Press 1~5 to select, ENTER to start", True, (220, 235, 255))
        screen.blit(inst, (400 - inst.get_width()//2, 360))
        pygame.display.flip()

    while selecting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = {pygame.K_1:1, pygame.K_2:2, pygame.K_3:3, pygame.K_4:4, pygame.K_5:5}[event.key]
                    selected_mode = [
                        GameEnv.MODE_CLASSIC,
                        GameEnv.MODE_POLAR,
                        GameEnv.MODE_TIMERALLY,
                        GameEnv.MODE_GATEBALL,
                        GameEnv.MODE_BEAT
                    ][idx-1]
                if event.key == pygame.K_RETURN:
                    selecting = False
        draw_title()
        clock.tick(60)

    # Launch env with selected local mode
    env = GameEnv(render_mode="human", local_mode=selected_mode)
    obs, info = env.reset()

    running = True
    while running:
        # Input
        action = 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and env.match_over:
                    obs, info = env.reset()
                elif event.key == pygame.K_e and env.local_mode == GameEnv.MODE_POLAR:
                    # Player polarity toggle
                    env.polar_player *= -1
                    env.polar_tint = 1.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 0
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2
        else:
            action = 1

        obs, rew, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            # 매치 종료 후 SPACE로 재시작
            pass

    env.close()
    pygame.quit()


# ---------------------------- Main ---------------------------- #

if __name__ == "__main__":
    run_human()
