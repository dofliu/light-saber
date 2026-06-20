"""
Star Wars Lightsaber v6 -- 多人鏡頭互動光劍 (順暢度強化)

v6 改動 (vs v5):
  - One Euro Filter 平滑 wrist / mid_mcp landmarks: 劍不再跟著手部
    landmark 抖動, 揮快時又不會拖慢 (專為 AR/VR 設計的自適應低通)
  - 偵測掉幀 velocity glide: mediapipe 漏一幀時用上次速度滑行 4 幀,
    避免劍瞬間消失閃爍, 比閉鎖更流暢
  - BBox-clipped glow rendering: 從整幀 copy 改成只 copy 劍周圍的
    bounding box, 4 把劍時渲染快 ~5x
  - Pre-warm mediapipe: 啟動時跑一張黑幀, 第一個真實偵測零延遲
  - 劍核心輕微脈衝: BLADE_CORE_WIDTH 用 sin(t*9) 微微震幅 0.5px,
    給 "活的能量" 感, 不是死的線

Install:
    pip install opencv-python mediapipe numpy pygame joblib scikit-learn

Run:
    python lightsaber_mvp.py

Keys:
    fist hold 0.5s   -> ignite
    open hand 0.3s   -> retract
    F                -> toggle fullscreen
    M                -> toggle mirror
    D                -> toggle landmark debug overlay
    1 / 2            -> shorter / longer blade
    ESC / Q          -> quit
"""

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


# -------- Tunables --------
MAX_HANDS = 4
BLADE_LEN_MULT = 7.0

BLADE_CORE_WIDTH = 10
BLADE_GLOW_WIDTH = 34
BLADE_HALO_WIDTH = 68
BLADE_WASH_WIDTH = 110
TRAIL_LEN = 6

FIST_CURL_RATIO = 1.05
FIST_MIN_CURLED_FINGERS = 3
OPEN_EXTEND_RATIO = 1.7
OPEN_MIN_EXTENDED_FINGERS = 4

FIST_HOLD_TIME = 0.5
OPEN_HOLD_TIME = 0.3
HAND_LOST_RETRACT_TIME = 0.5    # 手不見超過此秒數就強制收劍 (防止殘留)

HILT_FADE_IN = 0.35
HILT_FADE_OUT = 0.30
IGNITION_DURATION = 0.60
RETRACT_DURATION = 0.32
BLADE_GATE = 0.95

# Smoothing (v6)
ONE_EURO_MIN_CUTOFF = 2.0   # Hz; lower = more smoothing at rest
ONE_EURO_BETA = 0.10        # higher = less smoothing during fast motion
ONE_EURO_D_CUTOFF = 1.0
GLIDE_FRAMES = 4            # how many lost frames to keep gliding
BASE_EMA = 0.35             # hand-size smoothing (lower = smoother)

# SVM (off by default; available for future multi-gesture demo)
USE_SVM_BY_DEFAULT = False
SVM_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "gestureHand"))
SVM_MODEL = os.path.join(SVM_DIR, "svm_model_V3.pkl")
SVM_SCALER = os.path.join(SVM_DIR, "scaler_V3.pkl")
SVM_LABELS = os.path.join(SVM_DIR, "labels_V3.txt")
SVM_FIST_LABEL = "Rock"
SVM_FIST_CONFIDENCE = 0.40

SPARK_RADIUS = 80
SPARK_COUNT = 26

MATCH_MAX_DIST = 200
SLOT_TIMEOUT = 0.5

MP_PROCESS_SCALE = 0.5
MP_MODEL_COMPLEXITY = 0
CAP_WIDTH = 960
CAP_HEIGHT = 540
DISPLAY_WIDTH = 1600
DISPLAY_HEIGHT = 900

SWING_THRESHOLD_NORM = 8.0
SWING_FULL_NORM = 22.0
SWING_COOLDOWN = 0.16
TIP_SPEED_EMA = 0.5

SAMPLE_RATE = 22050

GAME_ROUND_SECONDS = 60.0
GAME_COUNTDOWN_SECONDS = 3.0
GAME_TARGET_LIFETIME = 2.8
GAME_COMBO_WINDOW = 2.2
GAME_MAX_TARGETS = 3
GAME_TARGET_RADIUS = 42
GAME_HIT_MIN_SPEED = 3.0
GAME_STARTING_LIVES = 3
GAME_LASER_RADIUS = 16


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    target_lifetime: float
    combo_window: float
    max_targets: int
    min_hit_speed: float
    direction_threshold: float
    direction_count: int
    laser_interval: float
    laser_speed: float


DIFFICULTY_PRESETS = {
    "easy": DifficultyPreset("easy", 4.0, 3.0, 2, 2.0, 0.15, 4, 5.5, 150.0),
    "normal": DifficultyPreset("normal", 2.8, 2.2, 3, 3.0, 0.45, 4, 3.8, 210.0),
    "hard": DifficultyPreset("hard", 1.9, 1.5, 4, 5.0, 0.70, 8, 2.4, 285.0),
}

SWING_DIRECTIONS = (
    ("RIGHT", np.array([1.0, 0.0], dtype=np.float32)),
    ("DOWN", np.array([0.0, 1.0], dtype=np.float32)),
    ("LEFT", np.array([-1.0, 0.0], dtype=np.float32)),
    ("UP", np.array([0.0, -1.0], dtype=np.float32)),
    ("DOWN-RIGHT", np.array([0.7071, 0.7071], dtype=np.float32)),
    ("DOWN-LEFT", np.array([-0.7071, 0.7071], dtype=np.float32)),
    ("UP-LEFT", np.array([-0.7071, -0.7071], dtype=np.float32)),
    ("UP-RIGHT", np.array([0.7071, -0.7071], dtype=np.float32)),
)

BLADE_COLORS = [
    (255,  95,  40),
    ( 50,  50, 255),
    ( 60, 255,  60),
    (210,  70, 210),
]
HILT_TONES = [
    (210, 210, 215),
    (155, 155, 160),
    (100, 100, 105),
    ( 60,  60,  65),
    ( 30,  30,  34),
]
HILT_DARK = (22, 22, 26)
HILT_BLACK = (10, 10, 12)


# -------- One Euro Filter --------
class OneEuroFilter:
    """Adaptive low-pass that smooths jitter at low speed but lets fast
    motion through. Used in many AR/VR systems for landmark stabilisation.
    See https://gery.casiez.net/1euro/ for the original paper."""
    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF,
                 beta=ONE_EURO_BETA, d_cutoff=ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.prev_x = None
        self.prev_dx = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x, dt):
        x = np.asarray(x, dtype=np.float32)
        if self.prev_x is None:
            self.prev_x = x.copy()
            self.prev_dx = np.zeros_like(x)
            return x.copy()
        dx = (x - self.prev_x) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        sdx = a_d * dx + (1 - a_d) * self.prev_dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(sdx))
        a = self._alpha(cutoff, dt)
        sx = a * x + (1 - a) * self.prev_x
        self.prev_x, self.prev_dx = sx, sdx
        return sx.copy()

    def reset(self):
        self.prev_x = None
        self.prev_dx = None


# -------- Helpers --------
def lm_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def to_pt(v):
    return (int(v[0]), int(v[1]))


def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# -------- Gesture detection --------
def is_fist_geom(lms, w, h):
    wrist = lm_xy(lms.landmark[0], w, h)
    curled = 0
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip = lm_xy(lms.landmark[tip_idx], w, h)
        mcp = lm_xy(lms.landmark[mcp_idx], w, h)
        if np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist) * FIST_CURL_RATIO:
            curled += 1
    return curled >= FIST_MIN_CURLED_FINGERS


def is_open_hand(lms, w, h):
    wrist = lm_xy(lms.landmark[0], w, h)
    extended = 0
    for tip_idx, mcp_idx in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip = lm_xy(lms.landmark[tip_idx], w, h)
        mcp = lm_xy(lms.landmark[mcp_idx], w, h)
        if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * OPEN_EXTEND_RATIO:
            extended += 1
    return extended >= OPEN_MIN_EXTENDED_FINGERS


_SVM_PAIRS = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
              (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16),
              (0, 17), (0, 18), (0, 19), (0, 20),
              (4, 8), (8, 12), (12, 16), (16, 20)]


def compute_svm_distances(lms):
    p_ref1 = np.array([lms.landmark[0].x, lms.landmark[0].y])
    p_ref2 = np.array([lms.landmark[9].x, lms.landmark[9].y])
    ref = float(np.linalg.norm(p_ref1 - p_ref2))
    if ref < 1e-6:
        return None
    out = []
    for a, b in _SVM_PAIRS:
        p1 = np.array([lms.landmark[a].x, lms.landmark[a].y])
        p2 = np.array([lms.landmark[b].x, lms.landmark[b].y])
        out.append(np.linalg.norm(p1 - p2) / ref)
    return out


class FistClassifier:
    def __init__(self):
        self.svm_ok = False
        self.use_svm = False
        self.clf = None
        self.scaler = None
        self.labels = None
        try:
            import joblib
            self.clf = joblib.load(SVM_MODEL)
            self.scaler = joblib.load(SVM_SCALER)
            with open(SVM_LABELS, "r", encoding="utf-8") as f:
                self.labels = [ln.strip() for ln in f if ln.strip()]
            if SVM_FIST_LABEL in self.labels:
                self.svm_ok = True
                self.use_svm = USE_SVM_BY_DEFAULT
                print("[SVM] available (%d labels). use_svm=%s"
                      % (len(self.labels), self.use_svm))
        except Exception as e:
            print("[SVM] not loaded (%s); geometric only." % e)

    def is_fist(self, lms, w, h):
        if not (self.use_svm and self.svm_ok):
            return is_fist_geom(lms, w, h)
        d = compute_svm_distances(lms)
        if d is None:
            return is_fist_geom(lms, w, h)
        try:
            X = self.scaler.transform([d])
            pred = int(self.clf.predict(X)[0])
            label = self.labels[pred] if 0 <= pred < len(self.labels) else None
            if label != SVM_FIST_LABEL:
                return False
            proba = float(np.max(self.clf.predict_proba(X)))
            return proba >= SVM_FIST_CONFIDENCE
        except Exception:
            return is_fist_geom(lms, w, h)


def segments_intersect(p1, p2, p3, p4):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
    return None


def segment_circle_hit(start, end, center, radius):
    """Return normalized hit accuracy, or None when the segment misses."""
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-6:
        closest = start
    else:
        projection = float(np.dot(center - start, segment) / length_sq)
        projection = max(0.0, min(1.0, projection))
        closest = start + segment * projection
    distance = float(np.linalg.norm(center - closest))
    if distance > radius:
        return None
    return max(0.0, 1.0 - distance / max(1.0, radius))


@dataclass(eq=False)
class GameTarget:
    target_id: int
    center: np.ndarray
    radius: int
    color: tuple
    direction_name: str
    direction: np.ndarray
    spawned_at: float
    expires_at: float
    last_wrong_at: float = -1e9


@dataclass(eq=False)
class LaserBolt:
    bolt_id: int
    origin: np.ndarray
    destination: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    radius: int
    color: tuple
    spawned_at: float
    danger_at: float


class ArcadeGame:
    """Camera-independent arcade round state and scoring logic."""

    def __init__(self, enabled=True, round_seconds=GAME_ROUND_SECONDS,
                 difficulty="normal", seed=None):
        self.enabled = enabled
        self.round_seconds = float(round_seconds)
        self.difficulty = DIFFICULTY_PRESETS[difficulty]
        self.rng = np.random.default_rng(seed)
        self.state = "ready" if enabled else "free"
        self.state_started_at = 0.0
        self.result_reason = ""
        self.round_started_at = 0.0
        self.result_reason = ""
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.hits = 0
        self.misses = 0
        self.parries = 0
        self.lives = GAME_STARTING_LIVES
        self.last_hit_at = -1e9
        self.targets = []
        self.laser_bolts = []
        self.hit_flashes = deque(maxlen=12)
        self._next_target_id = 1
        self._next_bolt_id = 1
        self.next_laser_at = 1e9

    def set_enabled(self, enabled, now):
        self.enabled = enabled
        self.reset(now)

    def reset(self, now):
        self.state = "ready" if self.enabled else "free"
        self.state_started_at = now
        self.round_started_at = 0.0
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.hits = 0
        self.misses = 0
        self.parries = 0
        self.lives = GAME_STARTING_LIVES
        self.last_hit_at = -1e9
        self.targets.clear()
        self.laser_bolts.clear()
        self.hit_flashes.clear()
        self.next_laser_at = 1e9

    def start(self, now):
        if not self.enabled or self.state not in ("ready", "results"):
            return False
        self.reset(now)
        self.state = "countdown"
        self.state_started_at = now
        return True

    def update(self, now, width, height):
        if not self.enabled:
            return
        if self.state == "countdown":
            if now - self.state_started_at >= GAME_COUNTDOWN_SECONDS:
                self.state = "playing"
                self.round_started_at = now
                self.state_started_at = now
                self.next_laser_at = now + self.difficulty.laser_interval
        if self.state != "playing":
            return
        if self.combo > 0 and now - self.last_hit_at > self.difficulty.combo_window:
            self.combo = 0
        if now - self.round_started_at >= self.round_seconds:
            self.state = "results"
            self.state_started_at = now
            self.result_reason = "complete"
            self.targets.clear()
            self.laser_bolts.clear()
            return

        active_bolts = []
        for bolt in self.laser_bolts:
            if now >= bolt.danger_at:
                self.misses += 1
                self.lives -= 1
                self.combo = 0
                self.hit_flashes.append(
                    (bolt.destination.copy(), now, 0, "SHIELD HIT"))
            else:
                bolt.position = bolt.origin + bolt.velocity * (now - bolt.spawned_at)
                active_bolts.append(bolt)
        self.laser_bolts = active_bolts
        if self.lives <= 0:
            self.state = "results"
            self.state_started_at = now
            self.result_reason = "game_over"
            self.targets.clear()
            self.laser_bolts.clear()
            return

        if now >= self.next_laser_at:
            self.laser_bolts.append(self._spawn_laser(now, width, height))
            jitter = float(self.rng.uniform(0.85, 1.15))
            self.next_laser_at = now + self.difficulty.laser_interval * jitter

        active_targets = []
        for target in self.targets:
            if now <= target.expires_at:
                active_targets.append(target)
            else:
                self.misses += 1
                self.combo = 0
        self.targets = active_targets

        desired = min(self.difficulty.max_targets, 1 + int(self.elapsed(now) // 15))
        while len(self.targets) < desired:
            self.targets.append(self._spawn_target(now, width, height))

    def _spawn_target(self, now, width, height):
        radius = max(26, min(GAME_TARGET_RADIUS, min(width, height) // 10))
        margin_x = radius + max(20, width // 12)
        top = radius + max(70, height // 10)
        bottom = max(top + 1, height - radius - max(60, height // 8))
        left = min(margin_x, max(radius, width - radius - 1))
        right = max(left + 1, width - margin_x)
        center = None
        for _ in range(12):
            candidate = np.array([
                self.rng.integers(left, right),
                self.rng.integers(top, bottom),
            ], dtype=np.float32)
            if all(np.linalg.norm(candidate - item.center) > radius * 2.5
                   for item in self.targets):
                center = candidate
                break
        if center is None:
            center = candidate
        direction_index = int(self.rng.integers(0, self.difficulty.direction_count))
        direction_name, direction = SWING_DIRECTIONS[direction_index]
        target = GameTarget(
            target_id=self._next_target_id,
            center=center,
            radius=radius,
            color=BLADE_COLORS[(self._next_target_id - 1) % len(BLADE_COLORS)],
            direction_name=direction_name,
            direction=direction.copy(),
            spawned_at=now,
            expires_at=now + self.difficulty.target_lifetime,
        )
        self._next_target_id += 1
        return target

    def _spawn_laser(self, now, width, height):
        edge = int(self.rng.integers(0, 4))
        if edge == 0:
            origin = np.array([0.0, self.rng.uniform(80, height - 80)], dtype=np.float32)
        elif edge == 1:
            origin = np.array([width - 1.0, self.rng.uniform(80, height - 80)], dtype=np.float32)
        elif edge == 2:
            origin = np.array([self.rng.uniform(80, width - 80), 0.0], dtype=np.float32)
        else:
            origin = np.array([self.rng.uniform(80, width - 80), height - 1.0], dtype=np.float32)
        destination = np.array([
            width * 0.5 + self.rng.uniform(-width * 0.10, width * 0.10),
            height * 0.55 + self.rng.uniform(-height * 0.10, height * 0.10),
        ], dtype=np.float32)
        travel = destination - origin
        distance = max(1.0, float(np.linalg.norm(travel)))
        velocity = travel / distance * self.difficulty.laser_speed
        duration = distance / self.difficulty.laser_speed
        bolt = LaserBolt(
            bolt_id=self._next_bolt_id,
            origin=origin,
            destination=destination,
            velocity=velocity,
            position=origin.copy(),
            radius=GAME_LASER_RADIUS,
            color=(60, 60, 255),
            spawned_at=now,
            danger_at=now + duration,
        )
        self._next_bolt_id += 1
        return bolt

    def register_blade(self, start, end, speed, now, swing_vector):
        if self.state != "playing" or speed < self.difficulty.min_hit_speed:
            return None
        swing_vector = np.asarray(swing_vector, dtype=np.float32)
        swing_length = float(np.linalg.norm(swing_vector))
        if swing_length <= 1e-6:
            return None
        swing_direction = swing_vector / swing_length
        for target in list(self.targets):
            accuracy = segment_circle_hit(start, end, target.center, target.radius)
            if accuracy is None:
                continue
            direction_score = float(np.dot(swing_direction, target.direction))
            if direction_score < self.difficulty.direction_threshold:
                if now - target.last_wrong_at > 0.35:
                    target.last_wrong_at = now
                    self.hit_flashes.append(
                        (target.center.copy(), now, 0, "WRONG WAY"))
                return None
            if now - self.last_hit_at <= self.difficulty.combo_window:
                self.combo += 1
            else:
                self.combo = 1
            self.best_combo = max(self.best_combo, self.combo)
            self.last_hit_at = now
            self.hits += 1
            speed_bonus = min(100, int(max(0.0, speed - SWING_THRESHOLD_NORM) * 5))
            accuracy_bonus = int(accuracy * 100)
            direction_bonus = int(max(0.0, direction_score) * 100)
            combo_multiplier = 1.0 + min(2.0, (self.combo - 1) * 0.1)
            gained = int(
                (100 + speed_bonus + accuracy_bonus + direction_bonus)
                * combo_multiplier)
            quality = "PERFECT" if (
                accuracy >= 0.72 and direction_score >= 0.85 and speed >= 10.0
            ) else "GOOD"
            self.score += gained
            self.targets.remove(target)
            self.hit_flashes.append((target.center.copy(), now, gained, quality))
            return target.center.copy(), gained
        return None

    def register_laser_parry(self, start, end, speed, now):
        if self.state != "playing" or speed < self.difficulty.min_hit_speed:
            return None
        for bolt in list(self.laser_bolts):
            accuracy = segment_circle_hit(
                start, end, bolt.position, bolt.radius + BLADE_CORE_WIDTH)
            if accuracy is None:
                continue
            gained = 150 + int(accuracy * 100)
            self.score += gained
            self.parries += 1
            self.laser_bolts.remove(bolt)
            self.hit_flashes.append(
                (bolt.position.copy(), now, gained, "PARRY"))
            return bolt.position.copy(), gained
        return None

    def elapsed(self, now):
        if self.round_started_at <= 0.0:
            return 0.0
        return max(0.0, now - self.round_started_at)

    def remaining(self, now):
        return max(0.0, self.round_seconds - self.elapsed(now))


# -------- Hilt --------
def hilt_geometry(wrist, mid_mcp, base):
    direction = (mid_mcp - wrist) / base
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)
    pommel = wrist - direction * (base * 0.40)
    emitter = mid_mcp + direction * (base * 0.55)
    half_w = base * 0.30
    return direction, perp, pommel, emitter, half_w


def _strip_poly(start, end, perp, half_w_top, half_w_bot):
    return np.array([
        start + perp * half_w_top, end + perp * half_w_top,
        end + perp * half_w_bot, start + perp * half_w_bot,
    ], dtype=np.int32)


def draw_hilt(frame, wrist, mid_mcp, base, color, alpha):
    if alpha <= 0:
        return
    direction, perp, pommel, emitter, half_w = hilt_geometry(wrist, mid_mcp, base)
    layer = frame.copy()

    n = len(HILT_TONES)
    for i, tone in enumerate(HILT_TONES):
        h_top = half_w * (1.0 - 2.0 * i / n)
        h_bot = half_w * (1.0 - 2.0 * (i + 1) / n)
        cv2.fillPoly(layer, [_strip_poly(pommel, emitter, perp, h_top, h_bot)], tone)

    pe_len = float(np.linalg.norm(emitter - pommel))
    pe_dir = (emitter - pommel) / max(1e-3, pe_len)
    for ratio in (0.20, 0.55, 0.85):
        pos = pommel + pe_dir * (pe_len * ratio)
        cv2.line(layer, to_pt(pos + perp * half_w), to_pt(pos - perp * half_w),
                 HILT_BLACK, 3, cv2.LINE_AA)

    grip_start = pommel + pe_dir * (pe_len * 0.22)
    grip_end = pommel + pe_dir * (pe_len * 0.53)
    grip_len = float(np.linalg.norm(grip_end - grip_start))
    n_x = 6
    for i in range(n_x):
        t = i / max(1, n_x - 1)
        p = grip_start * (1 - t) + grip_end * t
        offs = grip_len / n_x * 0.35
        cv2.line(layer, to_pt(p + perp * half_w * 0.92 - pe_dir * offs),
                 to_pt(p - perp * half_w * 0.92 + pe_dir * offs),
                 HILT_DARK, 1, cv2.LINE_AA)
        cv2.line(layer, to_pt(p + perp * half_w * 0.92 + pe_dir * offs),
                 to_pt(p - perp * half_w * 0.92 - pe_dir * offs),
                 HILT_DARK, 1, cv2.LINE_AA)

    body_start = pommel + pe_dir * (pe_len * 0.58)
    body_end = pommel + pe_dir * (pe_len * 0.82)
    n_riv = 4
    for i in range(n_riv):
        t = i / max(1, n_riv - 1)
        p = body_start * (1 - t) + body_end * t
        for side in (0.55, -0.55):
            rivet_pos = p + perp * (half_w * side)
            cv2.circle(layer, to_pt(rivet_pos), max(2, int(half_w * 0.10)),
                       HILT_BLACK, -1, cv2.LINE_AA)
            cv2.circle(layer, to_pt(rivet_pos), max(2, int(half_w * 0.10)),
                       HILT_TONES[1], 1, cv2.LINE_AA)

    cv2.line(layer, to_pt(emitter + perp * half_w),
             to_pt(emitter - perp * half_w), HILT_TONES[0], 4, cv2.LINE_AA)
    e_inner = emitter - direction * (base * 0.10)
    cv2.line(layer, to_pt(e_inner + perp * half_w * 0.95),
             to_pt(e_inner - perp * half_w * 0.95), HILT_TONES[2], 2, cv2.LINE_AA)
    e_glow = emitter - direction * (base * 0.06)
    cv2.line(layer, to_pt(e_glow + perp * half_w * 0.92),
             to_pt(e_glow - perp * half_w * 0.92), color, 3, cv2.LINE_AA)

    cv2.circle(layer, to_pt(pommel), max(3, int(half_w * 0.62)),
               HILT_BLACK, -1, cv2.LINE_AA)
    cv2.circle(layer, to_pt(pommel), max(2, int(half_w * 0.62)),
               HILT_TONES[0], 1, cv2.LINE_AA)
    cv2.circle(layer, to_pt(pommel), max(1, int(half_w * 0.30)),
               HILT_TONES[2], -1, cv2.LINE_AA)

    btn_pos = pommel + pe_dir * (pe_len * 0.70) + perp * (half_w * 0.55)
    btn_r = max(2, int(half_w * 0.22))
    cv2.circle(layer, to_pt(btn_pos), btn_r + 1, HILT_BLACK, -1, cv2.LINE_AA)
    cv2.circle(layer, to_pt(btn_pos), btn_r, color, -1, cv2.LINE_AA)
    cv2.circle(layer, to_pt(btn_pos), btn_r, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)


# -------- Charging visual --------
def draw_charging(frame, wrist, mid_mcp, base, color, progress, t):
    if progress <= 0:
        return
    p = max(0.0, min(1.0, progress))
    center = (wrist + mid_mcp) * 0.5
    pulse = 0.55 + 0.45 * math.sin(t * 14)
    intensity = (0.25 + 0.55 * p) * pulse

    r_outer = int(base * (3.0 - 2.4 * p))
    r_inner = int(base * 0.75)

    overlay = frame.copy()
    cv2.circle(overlay, to_pt(center), r_outer, color, 3, cv2.LINE_AA)
    cv2.circle(overlay, to_pt(center), r_inner, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, intensity, frame, 1 - intensity, 0, dst=frame)

    n_arcs = 5
    arc_span = 30
    for i in range(n_arcs):
        base_ang = (t * 90 + i * (360 / n_arcs)) % 360
        cv2.ellipse(frame, to_pt(center), (r_outer, r_outer),
                    0, base_ang, base_ang + arc_span,
                    (255, 255, 255), 2, cv2.LINE_AA)

    end_ang = int(360 * p)
    cv2.ellipse(frame, to_pt(center), (r_inner, r_inner),
                -90, 0, end_ang, (255, 255, 255), 3, cv2.LINE_AA)


# -------- Blade / trail / spark (v6: bbox-clipped glow) --------
def _glow_line(frame, p1, p2, color, thickness, alpha):
    """Alpha-blend a thick AA line, only copying the affected bbox region.
    Eliminates per-frame full-image copies (the v5 hot path)."""
    H, W = frame.shape[:2]
    pad = thickness + 2
    x_min = max(0, min(p1[0], p2[0]) - pad)
    y_min = max(0, min(p1[1], p2[1]) - pad)
    x_max = min(W, max(p1[0], p2[0]) + pad)
    y_max = min(H, max(p1[1], p2[1]) + pad)
    if x_max <= x_min or y_max <= y_min:
        return
    region = frame[y_min:y_max, x_min:x_max]
    overlay = region.copy()
    cv2.line(overlay,
             (p1[0] - x_min, p1[1] - y_min),
             (p2[0] - x_min, p2[1] - y_min),
             color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


def draw_blade(frame, start, end, color, core_w_extra=0.0):
    p1, p2 = to_pt(start), to_pt(end)
    _glow_line(frame, p1, p2, color, BLADE_WASH_WIDTH, 0.18)
    _glow_line(frame, p1, p2, color, BLADE_HALO_WIDTH, 0.40)
    _glow_line(frame, p1, p2, color, BLADE_GLOW_WIDTH, 0.60)
    core_w = max(2, int(BLADE_CORE_WIDTH + core_w_extra))
    cv2.line(frame, p1, p2, (255, 255, 255), core_w, cv2.LINE_AA)


def draw_tip_flash(frame, end, color, intensity):
    if intensity <= 0:
        return
    intensity = min(1.0, intensity)
    overlay = frame.copy()
    cv2.circle(overlay, to_pt(end), int(28 + 18 * intensity), color, -1, cv2.LINE_AA)
    a = 0.65 * intensity
    cv2.addWeighted(overlay, a, frame, 1 - a, 0, dst=frame)
    cv2.circle(frame, to_pt(end), int(10 + 8 * intensity), (255, 255, 255), -1, cv2.LINE_AA)


def draw_trail(frame, trail, color):
    items = list(trail)
    n = len(items)
    if n <= 1:
        return
    for k, (s, e) in enumerate(items[:-1]):
        alpha = (k + 1) / n * 0.35
        _glow_line(frame, to_pt(s), to_pt(e), color, BLADE_GLOW_WIDTH, alpha)


def draw_sparks(frame, center, seed):
    cx, cy = center
    rng = np.random.default_rng(seed & 0xFFFF)
    for _ in range(SPARK_COUNT):
        ang = rng.uniform(0, 2 * math.pi)
        r1 = rng.uniform(8, 22)
        r2 = rng.uniform(34, SPARK_RADIUS)
        cv2.line(frame,
                 (int(cx + math.cos(ang) * r1), int(cy + math.sin(ang) * r1)),
                 (int(cx + math.cos(ang) * r2), int(cy + math.sin(ang) * r2)),
                 (200, 240, 255), 2, cv2.LINE_AA)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), 70, (200, 240, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    cv2.circle(frame, (cx, cy), 24, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_centered_text(frame, text, y, scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (frame.shape[1] - text_width) // 2)
    cv2.putText(frame, text, (x + 2, y + 2), font, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale,
                color, thickness, cv2.LINE_AA)


def draw_arcade_targets(frame, game, now):
    if game.state != "playing":
        return
    for target in game.targets:
        center = to_pt(target.center)
        remaining_ratio = max(0.0, min(
            1.0,
            (target.expires_at - now) / game.difficulty.target_lifetime,
        ))
        pulse = 1.0 + 0.08 * math.sin(now * 9.0 + target.target_id)
        radius = max(4, int(target.radius * pulse))
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + 16, target.color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, dst=frame)
        cv2.circle(frame, center, radius, target.color, 5, cv2.LINE_AA)
        cv2.circle(frame, center, max(5, radius // 3), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.ellipse(frame, center, (radius + 8, radius + 8), -90,
                    0, int(360 * remaining_ratio),
                    (255, 255, 255), 4, cv2.LINE_AA)
        arrow_half = target.direction * (radius * 0.72)
        arrow_start = to_pt(target.center - arrow_half)
        arrow_end = to_pt(target.center + arrow_half)
        cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 255, 255),
                        4, cv2.LINE_AA, tipLength=0.32)


def draw_laser_bolts(frame, game, now):
    if game.state != "playing":
        return
    for bolt in game.laser_bolts:
        speed = max(1.0, float(np.linalg.norm(bolt.velocity)))
        direction = bolt.velocity / speed
        trail_end = bolt.position - direction * 70.0
        _glow_line(frame, to_pt(trail_end), to_pt(bolt.position),
                   bolt.color, 28, 0.42)
        cv2.line(frame, to_pt(trail_end), to_pt(bolt.position),
                 (255, 210, 210), 5, cv2.LINE_AA)
        pulse = 1.0 + 0.25 * math.sin(now * 18.0 + bolt.bolt_id)
        cv2.circle(frame, to_pt(bolt.position), int(bolt.radius * pulse),
                   bolt.color, -1, cv2.LINE_AA)
        cv2.circle(frame, to_pt(bolt.position), max(3, bolt.radius // 3),
                   (255, 255, 255), -1, cv2.LINE_AA)


def draw_arcade_ui(frame, game, now):
    if not game.enabled:
        cv2.putText(frame, "FREE PLAY  |  G: arcade mode", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        return

    if game.state == "playing":
        cv2.putText(frame, f"SCORE {game.score:06d}", (14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        combo_color = (80, 255, 255) if game.combo >= 5 else (220, 220, 220)
        cv2.putText(frame, f"COMBO x{game.combo}", (14, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, combo_color, 2, cv2.LINE_AA)
        cv2.putText(frame, game.difficulty.name.upper(), (14, 103),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
        lives_color = (80, 255, 80) if game.lives > 1 else (60, 60, 255)
        cv2.putText(frame, f"SHIELD {game.lives}", (14, 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lives_color, 2, cv2.LINE_AA)
        remaining_text = f"TIME {game.remaining(now):04.1f}"
        (text_width, _), _ = cv2.getTextSize(
            remaining_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, remaining_text, (frame.shape[1] - text_width - 14, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    elif game.state == "ready":
        _draw_centered_text(frame, "LIGHTSABER ARCADE", frame.shape[0] // 2 - 30,
                            1.15, (80, 255, 255), 3)
        _draw_centered_text(frame, "SPACE: start  |  G: free play",
                            frame.shape[0] // 2 + 25, 0.7, (255, 255, 255), 2)
    elif game.state == "countdown":
        remaining = max(1, int(math.ceil(
            GAME_COUNTDOWN_SECONDS - (now - game.state_started_at))))
        _draw_centered_text(frame, str(remaining), frame.shape[0] // 2,
                            2.8, (80, 255, 255), 5)
    elif game.state == "results":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, dst=frame)
        center_y = frame.shape[0] // 2
        game_over = game.result_reason == "game_over"
        result_title = "GAME OVER" if game_over else "ROUND COMPLETE"
        result_color = (60, 90, 255) if game_over else (80, 255, 255)
        _draw_centered_text(frame, result_title, center_y - 100,
                            1.2, result_color, 3)
        _draw_centered_text(frame, f"SCORE  {game.score}", center_y - 35,
                            1.0, (255, 255, 255), 2)
        _draw_centered_text(frame, f"HITS {game.hits}   MISSES {game.misses}",
                            center_y + 10, 0.75, (230, 230, 230), 2)
        _draw_centered_text(frame, f"PARRIES {game.parries}   BEST COMBO x{game.best_combo}",
                            center_y + 50, 0.75, (230, 230, 230), 2)
        _draw_centered_text(frame, "SPACE: play again  |  G: free play",
                            center_y + 110, 0.62, (255, 255, 255), 2)

    for center, hit_at, gained, quality in list(game.hit_flashes):
        age = now - hit_at
        if age > 0.7:
            continue
        alpha = 1.0 - age / 0.7
        pos = to_pt(center + np.array([0.0, -45.0 * age], dtype=np.float32))
        color = tuple(int(255 * alpha) for _ in range(3))
        cv2.putText(frame, f"{quality} +{gained}", pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2, cv2.LINE_AA)


# -------- Audio synthesis (unchanged from v5) --------
def _synth_ignition(duration=0.6):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    sub_env = np.exp(-t * 6) * (1 - np.exp(-t * 80))
    sub = np.sin(2 * np.pi * 45 * t) * 0.65 * sub_env
    f_curve = 50 + 240 * (t / duration) ** 0.7
    phase = 2 * np.pi * np.cumsum(f_curve) / SAMPLE_RATE
    main = np.sin(phase) * 0.40
    main += np.sin(phase * 1.005) * 0.25
    main += np.sin(phase * 0.995) * 0.25
    hiss = np.random.randn(n) * 0.40 * np.exp(-t * 5)
    ring = np.sin(2 * np.pi * 800 * t) * 0.18 * np.exp(-t * 12)
    env = np.minimum(1.0, t / 0.03) * (np.exp(-t * 1.0) * 0.65 + 0.35 * np.exp(-t * 0.4))
    return (sub + main + hiss + ring) * env


def _synth_retract(duration=0.5):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    f_curve = 280 - 250 * (t / duration) ** 0.6
    phase = 2 * np.pi * np.cumsum(f_curve) / SAMPLE_RATE
    main = np.sin(phase) * 0.40
    main += np.sin(phase * 1.005) * 0.22
    main += np.sin(phase * 0.995) * 0.22
    hiss = np.random.randn(n) * 0.28 * np.exp(-t * 4)
    click_t = duration - 0.06
    click = np.zeros(n)
    click_idx = int(click_t * SAMPLE_RATE)
    if click_idx < n:
        local_t = t[click_idx:] - click_t
        click[click_idx:] = np.exp(-local_t * 100) * np.sin(2 * np.pi * 500 * local_t) * 0.45
    env = np.exp(-t * 2.2)
    return (main + hiss + click) * env


def _synth_swing(duration=0.40):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    bell = np.sin(np.pi * t / duration)
    f = 180 + 800 * bell
    phase = 2 * np.pi * np.cumsum(f) / SAMPLE_RATE
    wav = np.sin(phase) * 0.35
    wav += np.sin(phase * 0.5) * 0.22
    wav += np.sin(phase * 1.005) * 0.20
    noise = np.random.randn(n) * 0.55
    env = bell ** 1.5
    return (wav + noise * 0.6) * env * 0.7


def _synth_clash(duration=0.40):
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.randn(n) * 0.65 * np.exp(-t * 6)
    f_main = 1500 - 1100 * (t / duration)
    main = np.sin(2 * np.pi * np.cumsum(f_main) / SAMPLE_RATE) * 0.40
    res = np.sin(2 * np.pi * 1800 * t) * 0.16 * np.exp(-t * 4)
    res += np.sin(2 * np.pi * 2350 * t) * 0.16 * np.exp(-t * 3.5)
    res += np.sin(2 * np.pi * 3100 * t) * 0.13 * np.exp(-t * 5)
    sub = np.sin(2 * np.pi * 60 * t) * 0.55 * np.exp(-t * 8)
    env = np.minimum(1.0, t / 0.005) * (np.exp(-t * 4.5) * 0.7 + 0.3 * np.exp(-t * 1))
    return (noise + main + res + sub) * env


class Audio:
    def __init__(self):
        self.ok = False
        self._last_swing_t = {}
        self._last_clash_t = 0.0
        try:
            import pygame
            self._pg = pygame
            pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
            pygame.mixer.init()
            self.s_ignite = self._mk(_synth_ignition())
            self.s_retract = self._mk(_synth_retract())
            self.s_swing = self._mk(_synth_swing())
            self.s_clash = self._mk(_synth_clash())
            self.ch_oneshot = pygame.mixer.Channel(0)
            self.ch_swing = pygame.mixer.Channel(1)
            self.ch_clash = pygame.mixer.Channel(2)
            self.ok = True
            print("[Audio] ok")
        except Exception as e:
            print("[Audio] disabled (%s)" % e)

    def _mk(self, arr):
        arr = np.clip(arr, -1.0, 1.0)
        arr16 = (arr * 32767.0).astype(np.int16)
        stereo = np.ascontiguousarray(np.stack([arr16, arr16], axis=-1))
        return self._pg.sndarray.make_sound(stereo)

    def play_ignite(self):
        if self.ok:
            self.ch_oneshot.play(self.s_ignite)

    def play_retract(self):
        if self.ok:
            self.ch_oneshot.play(self.s_retract)

    def play_clash(self, now):
        if not self.ok or now - self._last_clash_t < 0.05:
            return
        self.ch_clash.play(self.s_clash)
        self._last_clash_t = now

    def play_swing(self, slot_idx, intensity, now):
        if not self.ok:
            return
        if intensity < 0.18:
            return
        last = self._last_swing_t.get(slot_idx, 0.0)
        if now - last < SWING_COOLDOWN:
            return
        self.ch_swing.play(self.s_swing)
        self.ch_swing.set_volume(min(1.0, max(0.25, intensity)))
        self._last_swing_t[slot_idx] = now


# -------- Hand slot tracker (v6: filtered + glide) --------
class HandSlot:
    def __init__(self, idx):
        self.idx = idx
        self.color = BLADE_COLORS[idx % len(BLADE_COLORS)]
        self.active = False
        # Raw last-seen position (used for matching across frames)
        self.wrist = None
        self.mid_mcp = None
        # Filtered position (used for drawing)
        self.wrist_f = None
        self.mid_mcp_f = None
        self.base_f = 0.0
        # Velocity for glide-on-loss
        self.wrist_vel = np.zeros(2, dtype=np.float32)
        self.mid_mcp_vel = np.zeros(2, dtype=np.float32)
        self.frames_lost = 0
        # Filters
        self.wrist_filter = OneEuroFilter()
        self.mid_mcp_filter = OneEuroFilter()
        # State
        self.last_seen = 0.0
        self.ignited = False
        self.fist_pending_t = 0.0
        self.open_pending_t = 0.0
        self.blade_progress = 0.0
        self.hilt_progress = 0.0
        self.trail = deque(maxlen=TRAIL_LEN)
        self.prev_tip = None
        self.prev_tip_t = 0.0
        self.tip_speed_norm = 0.0
        self.tip_speed_smooth = 0.0
        self.tip_velocity_norm = np.zeros(2, dtype=np.float32)
        self.tip_velocity_smooth = np.zeros(2, dtype=np.float32)
        self.evt_ignite = False
        self.evt_retract = False


def assign_detections_to_slots(slots, detections):
    matches = {}
    used = set()
    for slot in slots:
        if not slot.active or slot.wrist is None:
            continue
        best, best_d = None, MATCH_MAX_DIST
        for di, (w, _, _, _, _) in enumerate(detections):
            if di in used:
                continue
            d = float(np.linalg.norm(slot.wrist - w))
            if d < best_d:
                best_d, best = d, di
        if best is not None:
            matches[slot.idx] = best
            used.add(best)
    for di in range(len(detections)):
        if di in used:
            continue
        for slot in slots:
            if not slot.active:
                slot.active = True
                matches[slot.idx] = di
                used.add(di)
                break
    return matches


def step_slot(slot, det, now, dt):
    slot.evt_ignite = False
    slot.evt_retract = False

    if det is not None:
        wrist_raw, mid_mcp_raw, base_raw, fist_now, open_now = det
        # Update velocity (from raw, before filtering)
        if slot.wrist is not None:
            slot.wrist_vel = (wrist_raw - slot.wrist) / max(dt, 1e-3)
            slot.mid_mcp_vel = (mid_mcp_raw - slot.mid_mcp) / max(dt, 1e-3)
        slot.wrist = wrist_raw
        slot.mid_mcp = mid_mcp_raw
        # Apply One Euro Filter
        slot.wrist_f = slot.wrist_filter.filter(wrist_raw, dt)
        slot.mid_mcp_f = slot.mid_mcp_filter.filter(mid_mcp_raw, dt)
        # EMA on hand size
        if slot.base_f <= 1e-3:
            slot.base_f = base_raw
        else:
            slot.base_f = (1 - BASE_EMA) * slot.base_f + BASE_EMA * base_raw
        slot.last_seen = now
        slot.frames_lost = 0
    else:
        # Detection lost — glide for a few frames using last velocity (damped)
        slot.frames_lost += 1
        if slot.frames_lost <= GLIDE_FRAMES and slot.wrist_f is not None:
            damp = 0.7 ** slot.frames_lost
            slot.wrist_f = slot.wrist_f + slot.wrist_vel * dt * damp
            slot.mid_mcp_f = slot.mid_mcp_f + slot.mid_mcp_vel * dt * damp
        else:
            slot.wrist_filter.reset()
            slot.mid_mcp_filter.reset()
        fist_now = False
        open_now = False

    # Auto-retract: 手消失超過閾值就強制收劍, 防止殘留 "鬼劍"
    if slot.ignited and (now - slot.last_seen) > HAND_LOST_RETRACT_TIME:
        slot.ignited = False
        slot.evt_retract = True
        slot.fist_pending_t = 0.0
        slot.open_pending_t = 0.0

    if not slot.ignited:
        if fist_now:
            slot.fist_pending_t = min(FIST_HOLD_TIME, slot.fist_pending_t + dt)
            if slot.fist_pending_t >= FIST_HOLD_TIME:
                slot.ignited = True
                slot.evt_ignite = True
                slot.fist_pending_t = 0.0
        else:
            slot.fist_pending_t = max(0.0, slot.fist_pending_t - dt * 2.0)
        slot.open_pending_t = 0.0
    else:
        if open_now:
            slot.open_pending_t = min(OPEN_HOLD_TIME, slot.open_pending_t + dt)
            if slot.open_pending_t >= OPEN_HOLD_TIME:
                slot.ignited = False
                slot.evt_retract = True
                slot.open_pending_t = 0.0
        else:
            slot.open_pending_t = max(0.0, slot.open_pending_t - dt * 2.0)
        slot.fist_pending_t = 0.0

    if slot.ignited:
        slot.hilt_progress = min(1.0, slot.hilt_progress + dt / HILT_FADE_IN)
        if slot.hilt_progress >= BLADE_GATE:
            slot.blade_progress = min(1.0, slot.blade_progress + dt / IGNITION_DURATION)
    else:
        slot.blade_progress = max(0.0, slot.blade_progress - dt / RETRACT_DURATION)
        if slot.blade_progress <= 0.001:
            slot.hilt_progress = max(0.0, slot.hilt_progress - dt / HILT_FADE_OUT)
        slot.trail.clear()
        slot.prev_tip = None
        slot.tip_speed_norm = 0.0
        slot.tip_speed_smooth = 0.0
        slot.tip_velocity_norm[:] = 0.0
        slot.tip_velocity_smooth[:] = 0.0

    if (now - slot.last_seen > SLOT_TIMEOUT
            and slot.blade_progress <= 0.001
            and slot.hilt_progress <= 0.001
            and slot.fist_pending_t <= 0.001):
        slot.active = False
        slot.wrist = None
        slot.wrist_f = None
        slot.mid_mcp_f = None
        slot.base_f = 0.0
        slot.wrist_filter.reset()
        slot.mid_mcp_filter.reset()
        slot.trail.clear()
        slot.ignited = False


def update_tip_speed(slot, tip_pos, now):
    if slot.prev_tip is not None and slot.prev_tip_t > 0 and slot.base_f > 1e-3:
        dt_t = max(1e-3, now - slot.prev_tip_t)
        d = float(np.linalg.norm(tip_pos - slot.prev_tip))
        slot.tip_speed_norm = (d / slot.base_f) / dt_t
        slot.tip_velocity_norm = ((tip_pos - slot.prev_tip) / slot.base_f) / dt_t
        slot.tip_speed_smooth = (slot.tip_speed_smooth * (1 - TIP_SPEED_EMA)
                                 + slot.tip_speed_norm * TIP_SPEED_EMA)
        slot.tip_velocity_smooth = (
            slot.tip_velocity_smooth * (1 - TIP_SPEED_EMA)
            + slot.tip_velocity_norm * TIP_SPEED_EMA)
    slot.prev_tip = tip_pos.copy()
    slot.prev_tip_t = now


# -------- Main --------
def parse_args(argv=None):
    def positive_int(value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be >= 1")
        return parsed

    def display_size(value):
        try:
            width_text, height_text = value.lower().split("x", 1)
            width = positive_int(width_text)
            height = positive_int(height_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "value must use WIDTHxHEIGHT format") from exc
        return width, height

    def process_scale(value):
        parsed = float(value)
        if parsed <= 0.0 or parsed > 1.0:
            raise argparse.ArgumentTypeError("value must be > 0 and <= 1")
        return parsed

    def model_complexity(value):
        parsed = int(value)
        if parsed not in (0, 1):
            raise argparse.ArgumentTypeError("value must be 0 or 1")
        return parsed

    def positive_float(value):
        parsed = float(value)
        if parsed <= 0.0:
            raise argparse.ArgumentTypeError("value must be > 0")
        return parsed

    parser = argparse.ArgumentParser(
        description="Lightsaber MVP with MediaPipe hand tracking.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index used by OpenCV VideoCapture (default: 0).")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring on preview.")
    parser.add_argument(
        "--max-hands",
        type=positive_int,
        default=MAX_HANDS,
        help="Maximum number of hands to track (default: %(default)s).")
    parser.add_argument(
        "--display-size",
        type=display_size,
        default=(DISPLAY_WIDTH, DISPLAY_HEIGHT),
        metavar="WIDTHxHEIGHT",
        help="Initial OpenCV window size (default: %(default)s).")
    parser.add_argument(
        "--process-scale",
        type=process_scale,
        default=MP_PROCESS_SCALE,
        help="Frame scale used before MediaPipe processing (default: %(default)s).")
    parser.add_argument(
        "--model-complexity",
        type=model_complexity,
        default=MP_MODEL_COMPLEXITY,
        help="MediaPipe Hands model complexity, 0 or 1 (default: %(default)s).")
    parser.add_argument(
        "--game-mode",
        choices=("arcade", "free"),
        default="arcade",
        help="Start in arcade or free-play mode (default: %(default)s).")
    parser.add_argument(
        "--difficulty",
        choices=tuple(DIFFICULTY_PRESETS),
        default="normal",
        help="Arcade difficulty preset (default: %(default)s).")
    parser.add_argument(
        "--round-seconds",
        type=positive_float,
        default=GAME_ROUND_SECONDS,
        help="Arcade round duration in seconds (default: %(default)s).")
    parser.add_argument(
        "--game-seed",
        type=int,
        default=None,
        help="Optional deterministic target seed for testing or demos.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        model_complexity=args.model_complexity,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    drawing_utils = mp.solutions.drawing_utils

    # Pre-warm mediapipe with a dummy frame so the first real detection has no init lag
    print("[Init] warming up mediapipe...")
    warmup = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
    warmup_rgb = cv2.cvtColor(warmup, cv2.COLOR_BGR2RGB)
    warmup_rgb.flags.writeable = False
    hands.process(warmup_rgb)
    print("[Init] ready")

    classifier = FistClassifier()
    audio = Audio()

    mirror = not args.no_mirror
    debug = False
    fullscreen = False
    blade_len_mult = BLADE_LEN_MULT

    win_name = "Lightsaber"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, *args.display_size)

    slots = [HandSlot(i) for i in range(args.max_hands)]
    game = ArcadeGame(
        enabled=args.game_mode == "arcade",
        round_seconds=args.round_seconds,
        difficulty=args.difficulty,
        seed=args.game_seed,
    )
    game.reset(time.time())
    last_t = time.time()
    fps_acc, fps_cnt, fps = 0.0, 0, 0.0

    print("[Lightsaber v6] hold fist 0.5s = ignite, open hand 0.3s = retract")
    print("                SPACE=start  G=game/free  R=reset  F=fullscreen")
    print("                M=mirror  D=debug  1/2=blade len  ESC/Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now
        game.update(now, w, h)

        if args.process_scale != 1.0:
            small = cv2.resize(frame, None, fx=args.process_scale, fy=args.process_scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)

        detections = []
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                wrist = lm_xy(lms.landmark[0], w, h)
                mid_mcp = lm_xy(lms.landmark[9], w, h)
                base = float(np.linalg.norm(mid_mcp - wrist))
                if base < 1e-3:
                    continue
                fist = classifier.is_fist(lms, w, h)
                openh = is_open_hand(lms, w, h)
                detections.append((wrist, mid_mcp, base, fist, openh))
                if debug:
                    drawing_utils.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

        matches = assign_detections_to_slots(slots, detections)
        for slot in slots:
            det = detections[matches[slot.idx]] if slot.idx in matches else None
            step_slot(slot, det, now, dt)
            if slot.evt_ignite:
                audio.play_ignite()
            if slot.evt_retract:
                audio.play_retract()

        draw_arcade_targets(frame, game, now)
        draw_laser_bolts(frame, game, now)

        # Render: use FILTERED positions (slot.wrist_f / mid_mcp_f / base_f)
        fully_ignited = []
        impact_hits = []
        peak_intensity = 0.0
        for slot in slots:
            if not slot.active or slot.wrist_f is None or slot.base_f <= 1e-3:
                continue

            if (not slot.ignited) and slot.fist_pending_t > 0.0:
                progress = slot.fist_pending_t / FIST_HOLD_TIME
                draw_charging(frame, slot.wrist_f, slot.mid_mcp_f, slot.base_f,
                              slot.color, progress, now)

            draw_hilt(frame, slot.wrist_f, slot.mid_mcp_f, slot.base_f,
                      slot.color, slot.hilt_progress)

            if slot.blade_progress > 0.001:
                _, _, _, emitter, _ = hilt_geometry(
                    slot.wrist_f, slot.mid_mcp_f, slot.base_f)
                direction = (slot.mid_mcp_f - slot.wrist_f) / slot.base_f
                eased = ease_out_cubic(slot.blade_progress)
                full_len = slot.base_f * blade_len_mult
                end = emitter + direction * (full_len * eased)

                if slot.blade_progress > 0.92:
                    slot.trail.append((emitter.copy(), end.copy()))
                    draw_trail(frame, slot.trail, slot.color)
                    update_tip_speed(slot, end, now)
                    intensity = max(0.0, (slot.tip_speed_smooth - SWING_THRESHOLD_NORM) /
                                    max(1e-3, SWING_FULL_NORM - SWING_THRESHOLD_NORM))
                    intensity = min(1.0, intensity)
                    if intensity > 0.0:
                        audio.play_swing(slot.idx, intensity, now)
                    if intensity > peak_intensity:
                        peak_intensity = intensity

                # v6: subtle pulse on core width for "alive energy" feel
                pulse = 0.5 * math.sin(now * 9.0 + slot.idx * 1.7)
                draw_blade(frame, emitter, end, slot.color, core_w_extra=pulse)

                if slot.blade_progress < 0.99:
                    draw_tip_flash(frame, end, slot.color, 1.0 - slot.blade_progress)
                else:
                    fully_ignited.append((emitter, end, slot.idx))
                    hit = game.register_blade(
                        emitter, end, slot.tip_speed_smooth, now,
                        slot.tip_velocity_smooth)
                    if hit is not None:
                        impact_hits.append(hit[0])
                    parry = game.register_laser_parry(
                        emitter, end, slot.tip_speed_smooth, now)
                    if parry is not None:
                        impact_hits.append(parry[0])

        for a in range(len(fully_ignited)):
            for b in range(a + 1, len(fully_ignited)):
                ip = segments_intersect(fully_ignited[a][0], fully_ignited[a][1],
                                        fully_ignited[b][0], fully_ignited[b][1])
                if ip is not None:
                    draw_sparks(frame, ip, int(now * 1000) + a * 17 + b)
                    audio.play_clash(now)

        for hit_center in impact_hits:
            draw_sparks(frame, to_pt(hit_center), int(now * 1000))
            audio.play_clash(now)

        fps_acc += dt
        fps_cnt += 1
        if fps_acc >= 0.5:
            fps = fps_cnt / fps_acc
            fps_acc, fps_cnt = 0.0, 0
        active = sum(1 for s in slots if s.active)
        ignited = sum(1 for s in slots if s.ignited)
        glide = sum(1 for s in slots if s.active and 0 < s.frames_lost <= GLIDE_FRAMES)
        hud = "v6  hands:%d  ignited:%d  glide:%d  blade x%.1f  swing:%.2f  %.1f fps" % (
            active, ignited, glide, blade_len_mult, peak_intensity, fps)
        cv2.putText(frame, hud, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
        draw_arcade_ui(frame, game, now)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord(' '):
            game.start(now)
        elif key in (ord('g'), ord('G')):
            game.set_enabled(not game.enabled, now)
        elif key in (ord('r'), ord('R')):
            game.reset(now)
        elif key in (ord('m'), ord('M')):
            mirror = not mirror
        elif key in (ord('d'), ord('D')):
            debug = not debug
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win_name, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('1'):
            blade_len_mult = max(2.0, blade_len_mult - 1.0)
        elif key == ord('2'):
            blade_len_mult = min(20.0, blade_len_mult + 1.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
