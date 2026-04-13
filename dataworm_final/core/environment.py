"""
environment.py
--------------
The DATA SPACE. This is the worm's world.
Not a grid of soil — a grid of data regions.

Each cell contains:
  - richness:  how much structured, learnable signal exists here (0.0–1.0)
  - novelty:   how unexplored this region is              (0.0–1.0, decays on visit)
  - danger:    how corrupt/noisy the data is              (0.0–1.0)
  - visited:   how many times the agent has been here

Think of it like a map of a dataset. Some regions are rich with patterns.
Some are noise. The worm navigates it using only its sensors — it cannot
see the whole map. It only feels what's immediately around it.
"""

import numpy as np
import json
import os
from datetime import datetime


class DataEnvironment:
    def __init__(self, width=40, height=40, seed=42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

        # --- Data space layers ---
        self.richness  = np.zeros((height, width))   # food equivalent
        self.danger    = np.zeros((height, width))   # poison equivalent
        self.novelty   = np.ones((height, width))    # starts fully novel
        self.visited   = np.zeros((height, width), dtype=int)
        self.reward_history = np.zeros((height, width))

        self._generate_world()

    def _generate_world(self):
        """
        Build a realistic data landscape.
        - Multiple food sources (rich data clusters)
        - Danger zones (corrupted/noisy regions)
        - Background noise everywhere
        """
        # Background: low richness everywhere
        self.richness = self.rng.uniform(0.0, 0.15, (self.height, self.width))

        # Place 4-6 rich data clusters (food sources)
        n_food = self.rng.integers(4, 7)
        self.food_centers = []
        for _ in range(n_food):
            cx = self.rng.integers(5, self.width - 5)
            cy = self.rng.integers(5, self.height - 5)
            strength = self.rng.uniform(0.6, 1.0)
            radius   = self.rng.uniform(3, 7)
            self.food_centers.append((cx, cy, strength, radius))
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        self.richness[y, x] = max(
                            self.richness[y, x],
                            strength * np.exp(-0.3 * dist)
                        )

        # Place 2-3 danger zones (corrupted data)
        n_danger = self.rng.integers(2, 4)
        self.danger_centers = []
        for _ in range(n_danger):
            cx = self.rng.integers(3, self.width - 3)
            cy = self.rng.integers(3, self.height - 3)
            strength = self.rng.uniform(0.7, 1.0)
            radius   = self.rng.uniform(2, 5)
            self.danger_centers.append((cx, cy, strength, radius))
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        self.danger[y, x] = max(
                            self.danger[y, x],
                            strength * np.exp(-0.4 * dist)
                        )

        # Clip everything to [0, 1]
        self.richness = np.clip(self.richness, 0, 1)
        self.danger   = np.clip(self.danger,   0, 1)

    def observe(self, x, y):
        """
        What the agent feels at position (x, y).
        Returns a dict of sensor readings — not the full map.
        The agent is blind to the whole. It only feels the local gradient.
        """
        def safe(arr, px, py):
            px = np.clip(px, 0, self.width  - 1)
            py = np.clip(py, 0, self.height - 1)
            return float(arr[py, px])

        return {
            # Left/right richness asymmetry (food sensing)
            'richness_L': safe(self.richness, x - 1, y),
            'richness_R': safe(self.richness, x + 1, y),
            'richness_F': safe(self.richness, x,     y - 1),

            # Left/right danger asymmetry
            'danger_L': safe(self.danger, x - 1, y),
            'danger_R': safe(self.danger, x + 1, y),
            'danger_F': safe(self.danger, x,     y - 1),

            # Novelty — how unexplored are adjacent cells
            'novelty_L': safe(self.novelty, x - 1, y),
            'novelty_R': safe(self.novelty, x + 1, y),
            'novelty_F': safe(self.novelty, x,     y - 1),

            # Current cell values
            'richness_here': safe(self.richness, x, y),
            'danger_here':   safe(self.danger,   x, y),
            'novelty_here':  safe(self.novelty,  x, y),
        }

    def step_into(self, x, y):
        """
        Agent moves into cell (x, y).
        Updates novelty (explored = less novel).
        Returns reward signal.
        """
        self.step_count += 1
        x = np.clip(x, 0, self.width  - 1)
        y = np.clip(y, 0, self.height - 1)

        self.visited[y, x] += 1

        # Novelty decays on visit — habituation
        self.novelty[y, x] = max(0.0, self.novelty[y, x] - 0.3)

        # Compute reward
        reward = (
            + self.richness[y, x] * 1.0    # food reward
            - self.danger[y, x]   * 2.0    # danger penalty (stronger)
            + self.novelty[y, x]  * 0.5    # curiosity bonus
        )
        self.reward_history[y, x] = reward

        return float(reward)

    def get_state_snapshot(self):
        """Full state for monitoring / logging."""
        return {
            'step': self.step_count,
            'richness': self.richness.tolist(),
            'danger':   self.danger.tolist(),
            'novelty':  self.novelty.tolist(),
            'visited':  self.visited.tolist(),
            'width':    self.width,
            'height':   self.height,
            'food_centers':   [(int(c[0]), int(c[1])) for c in self.food_centers],
            'danger_centers': [(int(c[0]), int(c[1])) for c in self.danger_centers],
        }
