"""
colony.py
---------
Manages a colony of independent worms in a shared world.

Each worm:
  - Has its OWN brain (no shared weights)
  - Makes its OWN decisions
  - Leaves pheromone trails that others can sense
  - Competes for the same food

The colony manager just handles turn order and logging.
It does NOT coordinate them. Any coordination must EMERGE.
"""

import numpy as np
from core.agent_v3 import DataWormV3
from core.environment_v4 import SocialEnvironment


class Colony:
    def __init__(self, n_worms, env_seed=42, pretrained_brain=None,
                 env_width=40, env_height=40):
        """
        n_worms: how many worms in the colony
        pretrained_brain: if provided, clone this worm's brain for all members
                          (simulates transfer from previous world)
        """
        self.n_worms = n_worms
        self.env = SocialEnvironment(width=env_width, height=env_height,
                                      seed=env_seed)

        # Spawn worms at different positions
        self.worms = []
        rng = np.random.default_rng(env_seed + 100)

        for i in range(n_worms):
            sx = rng.integers(2, env_width - 2)
            sy = rng.integers(2, env_height - 2)
            worm = DataWormV3(start_x=sx, start_y=sy,
                               env_width=env_width, env_height=env_height)
            worm.worm_id = i

            # If pretrained, copy brain weights
            if pretrained_brain is not None:
                worm.nerve_ring.W1 = pretrained_brain.nerve_ring.W1.copy()
                worm.nerve_ring.W2 = pretrained_brain.nerve_ring.W2.copy()
                worm.nerve_ring.b1 = pretrained_brain.nerve_ring.b1.copy()
                worm.nerve_ring.b2 = pretrained_brain.nerve_ring.b2.copy()
                worm.nerve_ring.n_inputs = pretrained_brain.nerve_ring.n_inputs
                worm.nerve_ring._trace_W1 = pretrained_brain.nerve_ring._trace_W1.copy()
                worm.nerve_ring._trace_W2 = pretrained_brain.nerve_ring._trace_W2.copy()
                worm.sensors.discovered_channels = list(
                    pretrained_brain.sensors.discovered_channels)
                worm.sensors.unknown_first_seen = pretrained_brain.sensors.unknown_first_seen

            self.worms.append(worm)

        self.step_count = 0
        self.colony_log = []

    def step(self):
        """One tick: all worms sense-decide-act, then global update."""
        self.step_count += 1
        step_data = []

        # Each worm takes a turn (randomized order to be fair)
        order = np.random.permutation(self.n_worms)

        for idx in order:
            worm = self.worms[idx]

            # Override the worm's sense method to pass worm_id
            raw_sensors = self.env.observe(worm.x, worm.y, worm_id=idx)

            # Use the worm's internal step but feed it our shared env
            entry = worm.step(self.env)

            # Make sure pheromone is deposited
            self.env.deposit_pheromone(worm.x, worm.y, idx)

            step_data.append({
                'worm_id': idx,
                'x': worm.x,
                'y': worm.y,
                'reward': entry['reward'],
                'action': entry['action'],
                'state': entry['state'],
                'found_unknown': entry.get('found_unknown', False),
            })

        # Global pheromone update
        self.env.global_step()

        self.colony_log.append(step_data)
        return step_data

    def run(self, n_steps, verbose_interval=100):
        """Run the colony for n_steps."""
        for step in range(n_steps):
            self.step()

            if verbose_interval and step % verbose_interval == 0 and step > 0:
                stats = self.get_stats()
                positions = [(w.x, w.y) for w in self.worms]
                pheromone_coverage = np.mean(self.env.pheromone > 0.01) * 100
                print(f"  Step {step:5d} | "
                      f"avg_reward {stats['avg_reward_recent']:+.3f} | "
                      f"pheromone_coverage {pheromone_coverage:.1f}% | "
                      f"total_food {stats['total_data_found']:.1f}")

    def get_stats(self):
        """Colony-wide statistics."""
        all_rewards = []
        for worm in self.worms:
            if worm.step_log:
                recent = worm.step_log[-20:]
                all_rewards.extend([s['reward'] for s in recent])

        # Spatial spread — are worms clustered or dispersed?
        positions = [(w.x, w.y) for w in self.worms]
        if len(positions) >= 2:
            dists = []
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    d = abs(positions[i][0]-positions[j][0]) + abs(positions[i][1]-positions[j][1])
                    dists.append(d)
            avg_spread = np.mean(dists)
        else:
            avg_spread = 0

        # How much food trail has each worm followed?
        pheromone_uses = [w.unknown_encounters for w in self.worms]

        return {
            'step': self.step_count,
            'n_worms': self.n_worms,
            'avg_reward_recent': float(np.mean(all_rewards)) if all_rewards else 0,
            'total_reward': sum(w.total_reward for w in self.worms),
            'total_data_found': sum(w.data_found for w in self.worms),
            'total_danger_hits': sum(w.danger_hits for w in self.worms),
            'avg_spread': float(avg_spread),
            'pheromone_coverage': float(np.mean(self.env.pheromone > 0.01) * 100),
            'pheromone_uses': pheromone_uses,
            'per_worm': [{
                'id': i,
                'pos': (w.x, w.y),
                'total_reward': round(w.total_reward, 2),
                'data_found': round(w.data_found, 2),
                'danger_hits': w.danger_hits,
                'unknown_encounters': w.unknown_encounters,
                'state': w.state,
                'n_inputs': w.nerve_ring.n_inputs,
            } for i, w in enumerate(self.worms)],
        }
