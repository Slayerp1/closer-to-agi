"""
environment_v4.py
-----------------
SOCIAL WORLD — shared environment with pheromone communication.

Multiple worms inhabit the same world.
They compete for food (eating depletes it).
They leave pheromone trails as they move.
Other worms sense these trails as unknown signals.

Nobody tells them to cooperate or compete.
The social structure EMERGES from individual decisions.
"""

import numpy as np


class SocialEnvironment:
    def __init__(self, width=40, height=40, seed=42, n_food=5, n_danger=2):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # World layers
        self.richness = np.zeros((height, width))
        self.danger = np.zeros((height, width))
        self.novelty = np.ones((height, width))
        self.visited = np.zeros((height, width), dtype=int)
        self.walls = np.zeros((height, width), dtype=bool)

        # PHEROMONE LAYER — the social signal
        # Each worm deposits pheromone as it moves
        # Other worms sense it through the unknown signal channel
        self.pheromone = np.zeros((height, width))
        self.PHEROMONE_DEPOSIT = 0.5    # how much a worm leaves per step
        self.PHEROMONE_DECAY = 0.985    # per-step global decay
        self.PHEROMONE_DIFFUSE = 0.03   # spread to neighbors per step

        # Track which worm is where (for collision/crowding)
        self.worm_positions = {}  # worm_id → (x, y)

        self.food_centers = []
        self.danger_centers = []
        self.events = []

        self._generate_world(n_food, n_danger)

    def _generate_world(self, n_food, n_danger):
        self.richness = self.rng.uniform(0.0, 0.12, (self.height, self.width))

        for _ in range(n_food):
            cx = self.rng.integers(4, self.width - 4)
            cy = self.rng.integers(4, self.height - 4)
            strength = self.rng.uniform(0.6, 1.0)
            radius = self.rng.uniform(3, 7)
            self.food_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.richness, cx, cy, strength, radius, 0.3)

        for _ in range(n_danger):
            cx = self.rng.integers(3, self.width - 3)
            cy = self.rng.integers(3, self.height - 3)
            strength = self.rng.uniform(0.5, 0.8)
            radius = self.rng.uniform(2, 4)
            self.danger_centers.append([cx, cy, strength, radius])
            self._paint_gaussian(self.danger, cx, cy, strength, radius, 0.4)

        self.richness = np.clip(self.richness, 0, 1)
        self.danger = np.clip(self.danger, 0, 1)

    def _paint_gaussian(self, layer, cx, cy, strength, radius, decay):
        for y in range(self.height):
            for x in range(self.width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius * 2.5:
                    layer[y, x] = max(layer[y, x], strength * np.exp(-decay * dist))

    def _update_pheromone(self):
        """Decay and diffuse pheromones. Called once per global step."""
        # Decay
        self.pheromone *= self.PHEROMONE_DECAY

        # Diffusion — each cell shares a bit with neighbors
        if self.PHEROMONE_DIFFUSE > 0:
            diffused = np.zeros_like(self.pheromone)
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                shifted = np.roll(np.roll(self.pheromone, dy, axis=0), dx, axis=1)
                diffused += shifted
            self.pheromone = (1 - self.PHEROMONE_DIFFUSE * 4) * self.pheromone + \
                             self.PHEROMONE_DIFFUSE * diffused

        self.pheromone = np.clip(self.pheromone, 0, 1)

    def deposit_pheromone(self, x, y, worm_id):
        """Worm leaves its scent at current position."""
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        self.pheromone[y, x] = min(1.0, self.pheromone[y, x] + self.PHEROMONE_DEPOSIT)
        self.worm_positions[worm_id] = (x, y)

    def count_nearby_worms(self, x, y, worm_id, radius=3):
        """How many other worms are nearby? For crowding detection."""
        count = 0
        for wid, (wx, wy) in self.worm_positions.items():
            if wid != worm_id:
                dist = abs(wx - x) + abs(wy - y)
                if dist <= radius:
                    count += 1
        return count

    def is_wall(self, x, y):
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        return bool(self.walls[y, x])

    def observe(self, x, y, worm_id=None):
        """What a specific worm senses. Pheromone = unknown signal."""
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
            # PHEROMONE as unknown signal — other worms' trails
            'unknown_L': safe(self.pheromone, x - 1, y),
            'unknown_R': safe(self.pheromone, x + 1, y),
            'unknown_F': safe(self.pheromone, x,     y - 1),
            'unknown_here': safe(self.pheromone, x, y),
        }

        return obs

    def step_into(self, x, y, worm_id=None):
        """A worm enters this cell. Shared food depletes for everyone."""
        self.step_count += 1
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        self.visited[y, x] += 1
        self.novelty[y, x] = max(0.0, self.novelty[y, x] - 0.2)

        # Food depletes when eaten — COMPETITIVE
        food_eaten = self.richness[y, x] * 0.05
        self.richness[y, x] = max(0, self.richness[y, x] - food_eaten)

        # Crowding penalty — too many worms in one spot is bad
        nearby = self.count_nearby_worms(x, y, worm_id) if worm_id is not None else 0
        crowding_penalty = nearby * 0.1

        reward = (
            + (food_eaten * 25)
            - self.danger[y, x] * 2.0
            + self.novelty[y, x] * 0.4
            - crowding_penalty
        )

        # Deposit pheromone — leave your mark
        if worm_id is not None:
            self.deposit_pheromone(x, y, worm_id)

        return float(reward)

    def global_step(self):
        """Called once per tick after all worms have moved. Updates pheromones."""
        self._update_pheromone()

    def get_state_snapshot(self):
        return {
            'step': self.step_count,
            'width': self.width,
            'height': self.height,
            'pheromone': self.pheromone.tolist(),
            'richness': self.richness.tolist(),
            'food_centers': [(int(c[0]), int(c[1])) for c in self.food_centers],
            'worm_positions': {str(k): v for k, v in self.worm_positions.items()},
        }
