"""
environment_v2.py
-----------------
THE HOSTILE, CHANGING WORLD.

Upgrades from v1:
  - DYNAMIC EVENTS: food moves, danger zones shift, walls appear mid-run
  - UNKNOWN SIGNALS: new signal types the worm has NEVER seen before
  - WALLS: impassable barriers that appear suddenly
  - FOOD DEPLETION: eating food reduces it (not infinite)
  - SEASONAL SHIFTS: the whole landscape changes periodically

The test: can the worm handle what it was never designed for?
"""

import numpy as np
import json


class DynamicEnvironment:
    def __init__(self, width=40, height=40, seed=42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # Core layers
        self.richness = np.zeros((height, width))
        self.danger   = np.zeros((height, width))
        self.novelty  = np.ones((height, width))
        self.visited  = np.zeros((height, width), dtype=int)
        self.walls    = np.zeros((height, width), dtype=bool)

        # UNKNOWN SIGNAL — a new dimension the worm has never encountered
        # Starts as all zeros. Appears mid-run.
        self.unknown_signal = np.zeros((height, width))
        self.unknown_active = False
        self.unknown_type = None  # will be assigned when activated

        # Event log — what happened when
        self.events = []

        self._generate_world()

    def _generate_world(self):
        """Build initial landscape."""
        self.richness = self.rng.uniform(0.0, 0.15, (self.height, self.width))

        # 5 food clusters
        n_food = 5
        self.food_centers = []
        for _ in range(n_food):
            cx = self.rng.integers(5, self.width - 5)
            cy = self.rng.integers(5, self.height - 5)
            strength = self.rng.uniform(0.6, 1.0)
            radius = self.rng.uniform(3, 7)
            self.food_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.richness, cx, cy, strength, radius, 0.3)

        # 2 danger zones
        n_danger = 2
        self.danger_centers = []
        for _ in range(n_danger):
            cx = self.rng.integers(3, self.width - 3)
            cy = self.rng.integers(3, self.height - 3)
            strength = self.rng.uniform(0.7, 1.0)
            radius = self.rng.uniform(2, 5)
            self.danger_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.danger, cx, cy, strength, radius, 0.4)

        self.richness = np.clip(self.richness, 0, 1)
        self.danger = np.clip(self.danger, 0, 1)

    def _paint_gaussian(self, layer, cx, cy, strength, radius, decay):
        for y in range(self.height):
            for x in range(self.width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius * 2:
                    layer[y, x] = max(layer[y, x], strength * np.exp(-decay * dist))

    # ─────────────────────────────────────────────
    # DYNAMIC EVENTS — the world fights back
    # ─────────────────────────────────────────────

    def _check_events(self):
        """Trigger events based on step count."""

        # EVENT 1 (step 300): Food source #0 dies. Depleted.
        if self.step_count == 300:
            self._event_kill_food(0)

        # EVENT 2 (step 500): Wall appears — blocks a corridor
        if self.step_count == 500:
            self._event_add_wall()

        # EVENT 3 (step 800): UNKNOWN SIGNAL appears
        # Something completely new. The worm has no sensor for this.
        # But it CORRELATES with food — will the worm learn the correlation?
        if self.step_count == 800:
            self._event_unknown_signal()

        # EVENT 4 (step 1200): Danger zone MOVES
        if self.step_count == 1200:
            self._event_move_danger()

        # EVENT 5 (step 1500): Food appears in a previously empty area
        if self.step_count == 1500:
            self._event_new_food()

        # EVENT 6 (step 1800): CATASTROPHE — all food reduced by 50%
        if self.step_count == 1800:
            self._event_catastrophe()

    def _event_kill_food(self, idx):
        if idx < len(self.food_centers):
            fc = self.food_centers[idx]
            cx, cy = fc[0], fc[1]
            radius = fc[2] if len(fc) > 3 else 5
            # Zero out food in that area
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius * 2:
                        self.richness[y, x] *= 0.1
            self.events.append({
                'step': self.step_count,
                'type': 'FOOD_DIED',
                'desc': f'Food source at ({cx},{cy}) depleted',
            })

    def _event_add_wall(self):
        # Place a wall across the middle-ish area
        wall_y = self.height // 2
        gap = self.rng.integers(5, self.width - 5)
        for x in range(self.width):
            if abs(x - gap) > 3:  # leave a gap of 6 cells
                self.walls[wall_y, x] = True
                self.walls[wall_y - 1, x] = True
        self.events.append({
            'step': self.step_count,
            'type': 'WALL_APPEARED',
            'desc': f'Wall at y={wall_y} with gap near x={gap}',
            'gap_x': int(gap),
        })

    def _event_unknown_signal(self):
        """
        A COMPLETELY NEW type of signal appears in the world.
        The worm has NO hardcoded sensor for it.
        But it spatially correlates with a new food source.
        The test: can the worm's adaptive sensors pick it up?
        """
        self.unknown_active = True
        self.unknown_type = 'beacon'  # like a pheromone trail

        # Place beacon near a new food location
        bx = self.rng.integers(25, 35)
        by = self.rng.integers(25, 35)
        self._paint_gaussian(self.unknown_signal, bx, by, 0.9, 8, 0.15)

        # Also place food there — the signal PREDICTS food
        self._paint_gaussian(self.richness, bx, by, 0.8, 5, 0.3)
        self.richness = np.clip(self.richness, 0, 1)
        self.unknown_signal = np.clip(self.unknown_signal, 0, 1)

        self.food_centers.append([bx, by, 0.8, 5])

        self.events.append({
            'step': self.step_count,
            'type': 'UNKNOWN_SIGNAL',
            'desc': f'New beacon signal at ({bx},{by}) — correlates with food. Worm has no sensor for this.',
            'beacon_pos': (int(bx), int(by)),
        })

    def _event_move_danger(self):
        if self.danger_centers:
            dc = self.danger_centers[0]
            old_x, old_y = dc[0], dc[1]
            # Clear old danger
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - old_x)**2 + (y - old_y)**2)
                    if dist < 8:
                        self.danger[y, x] *= 0.1

            # New location
            new_x = self.rng.integers(10, 30)
            new_y = self.rng.integers(10, 30)
            dc[0], dc[1] = new_x, new_y
            self._paint_gaussian(self.danger, new_x, new_y, dc[2], dc[3], 0.4)
            self.danger = np.clip(self.danger, 0, 1)

            self.events.append({
                'step': self.step_count,
                'type': 'DANGER_MOVED',
                'desc': f'Danger zone moved from ({old_x},{old_y}) to ({new_x},{new_y})',
            })

    def _event_new_food(self):
        cx = self.rng.integers(20, 35)
        cy = self.rng.integers(5, 15)
        self._paint_gaussian(self.richness, cx, cy, 0.9, 6, 0.25)
        self.richness = np.clip(self.richness, 0, 1)
        self.food_centers.append([cx, cy, 0.9, 6])
        self.events.append({
            'step': self.step_count,
            'type': 'NEW_FOOD',
            'desc': f'New food source appeared at ({cx},{cy})',
        })

    def _event_catastrophe(self):
        self.richness *= 0.5
        self.events.append({
            'step': self.step_count,
            'type': 'CATASTROPHE',
            'desc': 'All food reduced by 50%. Famine.',
        })

    # ─────────────────────────────────────────────
    # CORE API
    # ─────────────────────────────────────────────

    def is_wall(self, x, y):
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        return bool(self.walls[y, x])

    def observe(self, x, y):
        """Sensor readings — what the worm feels."""
        def safe(arr, px, py):
            px = np.clip(px, 0, self.width - 1)
            py = np.clip(py, 0, self.height - 1)
            return float(arr[py, px])

        def wall_check(px, py):
            px = np.clip(px, 0, self.width - 1)
            py = np.clip(py, 0, self.height - 1)
            return 1.0 if self.walls[py, px] else 0.0

        obs = {
            'richness_L': safe(self.richness, x - 1, y),
            'richness_R': safe(self.richness, x + 1, y),
            'richness_F': safe(self.richness, x,     y - 1),

            'danger_L': safe(self.danger, x - 1, y),
            'danger_R': safe(self.danger, x + 1, y),
            'danger_F': safe(self.danger, x,     y - 1),

            'novelty_L': safe(self.novelty, x - 1, y),
            'novelty_R': safe(self.novelty, x + 1, y),
            'novelty_F': safe(self.novelty, x,     y - 1),

            'richness_here': safe(self.richness, x, y),
            'danger_here':   safe(self.danger,   x, y),
            'novelty_here':  safe(self.novelty,  x, y),

            # Wall sensing — new but fits existing sensor pattern
            'wall_L': wall_check(x - 1, y),
            'wall_R': wall_check(x + 1, y),
            'wall_F': wall_check(x,     y - 1),
        }

        # Unknown signal — raw, unprocessed
        # The worm doesn't know what this IS. It's just... something.
        if self.unknown_active:
            obs['unknown_L'] = safe(self.unknown_signal, x - 1, y)
            obs['unknown_R'] = safe(self.unknown_signal, x + 1, y)
            obs['unknown_F'] = safe(self.unknown_signal, x,     y - 1)
            obs['unknown_here'] = safe(self.unknown_signal, x, y)
        else:
            obs['unknown_L'] = 0.0
            obs['unknown_R'] = 0.0
            obs['unknown_F'] = 0.0
            obs['unknown_here'] = 0.0

        return obs

    def step_into(self, x, y):
        """Agent enters cell. Returns reward. Triggers events."""
        self.step_count += 1
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Check for dynamic events
        self._check_events()

        self.visited[y, x] += 1
        self.novelty[y, x] = max(0.0, self.novelty[y, x] - 0.3)

        # Food depletes slightly when eaten (not infinite)
        food_eaten = self.richness[y, x] * 0.05
        self.richness[y, x] = max(0, self.richness[y, x] - food_eaten)

        reward = (
            + (food_eaten * 20)              # food reward (scaled up since it depletes)
            - self.danger[y, x] * 2.0        # danger penalty
            + self.novelty[y, x] * 0.5       # curiosity bonus
        )

        # Bonus for unknown signal — the worm doesn't know WHY
        # but being near the beacon feels slightly good
        if self.unknown_active:
            reward += self.unknown_signal[y, x] * 0.3

        self.reward_history = getattr(self, 'reward_history',
                                       np.zeros((self.height, self.width)))
        self.reward_history[y, x] = reward

        return float(reward)

    def get_state_snapshot(self):
        return {
            'step': self.step_count,
            'richness': self.richness.tolist(),
            'danger': self.danger.tolist(),
            'novelty': self.novelty.tolist(),
            'visited': self.visited.tolist(),
            'walls': self.walls.tolist(),
            'unknown_signal': self.unknown_signal.tolist(),
            'unknown_active': self.unknown_active,
            'width': self.width,
            'height': self.height,
            'food_centers': [(int(c[0]), int(c[1])) for c in self.food_centers],
            'danger_centers': [(int(c[0]), int(c[1])) for c in self.danger_centers],
            'events': self.events,
        }
