"""
environment_v3.py
-----------------
TRANSFER-CAPABLE WORLD.

Supports multiple configurable signal types:
  - 'glow'      — spatial beacon near food (like v2's unknown signal)
  - 'vibration' — pulsing signal that strengthens near food (different pattern)
  - 'heat'      — gradient that radiates from food sources directly

Each signal PREDICTS food but in a different spatial pattern.
The worm must learn the association from scratch each time.

The meta-learning test: does learning one signal→food mapping
make learning the NEXT one faster?
"""

import numpy as np


class TransferEnvironment:
    def __init__(self, width=40, height=40, seed=42,
                 signal_type=None, signal_on_step=0):
        """
        signal_type: 'glow', 'vibration', 'heat', or None
        signal_on_step: when the signal activates (0 = from start)
        """
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        self.richness = np.zeros((height, width))
        self.danger = np.zeros((height, width))
        self.novelty = np.ones((height, width))
        self.visited = np.zeros((height, width), dtype=int)
        self.walls = np.zeros((height, width), dtype=bool)

        # Signal system
        self.signal_type = signal_type
        self.signal_on_step = signal_on_step
        self.signal_layer = np.zeros((height, width))
        self.signal_active = (signal_on_step == 0 and signal_type is not None)

        self.events = []
        self.food_centers = []
        self.danger_centers = []

        self._generate_world()

        if self.signal_active:
            self._generate_signal()

    def _generate_world(self):
        self.richness = self.rng.uniform(0.0, 0.15, (self.height, self.width))

        n_food = self.rng.integers(4, 7)
        for _ in range(n_food):
            cx = self.rng.integers(5, self.width - 5)
            cy = self.rng.integers(5, self.height - 5)
            strength = self.rng.uniform(0.6, 1.0)
            radius = self.rng.uniform(3, 7)
            self.food_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.richness, cx, cy, strength, radius, 0.3)

        n_danger = self.rng.integers(1, 3)
        for _ in range(n_danger):
            cx = self.rng.integers(3, self.width - 3)
            cy = self.rng.integers(3, self.height - 3)
            strength = self.rng.uniform(0.5, 0.8)
            radius = self.rng.uniform(2, 4)
            self.danger_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.danger, cx, cy, strength, radius, 0.4)

        self.richness = np.clip(self.richness, 0, 1)
        self.danger = np.clip(self.danger, 0, 1)

    def _generate_signal(self):
        """Generate the predictive signal based on type."""
        self.signal_layer = np.zeros((self.height, self.width))

        if self.signal_type == 'glow':
            # Beacon — concentrated hotspot near the richest food
            # Same as v2's unknown signal
            best = max(self.food_centers, key=lambda c: c[2])
            cx, cy = best[0], best[1]
            # Slight offset — signal is NEAR food, not exactly on it
            ox = self.rng.integers(-3, 4)
            oy = self.rng.integers(-3, 4)
            self._paint_gaussian(self.signal_layer, cx+ox, cy+oy, 0.9, 8, 0.15)

        elif self.signal_type == 'vibration':
            # Distributed — multiple weak pulses near ALL food sources
            # Different spatial pattern than glow
            for fc in self.food_centers:
                cx, cy = fc[0], fc[1]
                # Vibration is wider, weaker, more spread out
                self._paint_gaussian(self.signal_layer, cx, cy, 0.4, 12, 0.08)

        elif self.signal_type == 'heat':
            # Gradient — directly proportional to richness but noisier
            # The signal IS the food gradient, but corrupted with noise
            self.signal_layer = self.richness * 0.7 + self.rng.uniform(0, 0.15, (self.height, self.width))

        self.signal_layer = np.clip(self.signal_layer, 0, 1)
        self.signal_active = True

        self.events.append({
            'step': self.step_count,
            'type': 'SIGNAL_ACTIVATED',
            'desc': f'Signal type "{self.signal_type}" now active',
        })

    def _paint_gaussian(self, layer, cx, cy, strength, radius, decay):
        for y in range(self.height):
            for x in range(self.width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius * 2.5:
                    layer[y, x] = max(layer[y, x], strength * np.exp(-decay * dist))

    def is_wall(self, x, y):
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        return bool(self.walls[y, x])

    def observe(self, x, y):
        def safe(arr, px, py):
            px = np.clip(px, 0, self.width - 1)
            py = np.clip(py, 0, self.height - 1)
            return float(arr[py, px])

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
            'wall_L': 1.0 if self.is_wall(x-1, y) else 0.0,
            'wall_R': 1.0 if self.is_wall(x+1, y) else 0.0,
            'wall_F': 1.0 if self.is_wall(x, y-1) else 0.0,
        }

        # Unknown/predictive signal — delivered as same channel name
        # regardless of type. The worm sees it as one "unknown" sense.
        if self.signal_active:
            obs['unknown_L'] = safe(self.signal_layer, x - 1, y)
            obs['unknown_R'] = safe(self.signal_layer, x + 1, y)
            obs['unknown_F'] = safe(self.signal_layer, x,     y - 1)
            obs['unknown_here'] = safe(self.signal_layer, x, y)
        else:
            obs['unknown_L'] = 0.0
            obs['unknown_R'] = 0.0
            obs['unknown_F'] = 0.0
            obs['unknown_here'] = 0.0

        return obs

    def step_into(self, x, y):
        self.step_count += 1
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Activate signal if it's time
        if (not self.signal_active and self.signal_type is not None
                and self.step_count >= self.signal_on_step):
            self._generate_signal()

        self.visited[y, x] += 1
        self.novelty[y, x] = max(0.0, self.novelty[y, x] - 0.3)

        food_eaten = self.richness[y, x] * 0.03
        self.richness[y, x] = max(0, self.richness[y, x] - food_eaten)

        reward = (
            + (food_eaten * 30)
            - self.danger[y, x] * 2.0
            + self.novelty[y, x] * 0.5
        )

        if self.signal_active:
            reward += self.signal_layer[y, x] * 0.2

        return float(reward)

    def get_state_snapshot(self):
        return {
            'step': self.step_count,
            'width': self.width,
            'height': self.height,
            'signal_type': self.signal_type,
            'signal_active': self.signal_active,
            'food_centers': [(int(c[0]), int(c[1])) for c in self.food_centers],
            'danger_centers': [(int(c[0]), int(c[1])) for c in self.danger_centers],
            'events': self.events,
        }
