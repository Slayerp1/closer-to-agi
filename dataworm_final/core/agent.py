"""
agent.py
--------
The worm. The complete agent.

It has:
  - A body (position in the data space)
  - Sensors (reads the environment)
  - A curiosity engine (innate drive — cannot be removed)
  - A nerve ring (learned integrator)
  - An avoidance reflex (hardwired safety — fires before thinking)
  - A memory (the synaptic weights themselves)

It does NOT have:
  - A goal given by a human
  - A dataset to train on
  - A prompt to respond to
  - A reward function designed by an engineer (the curiosity IS the reward)

It lives. It explores. It learns by moving.
"""

import numpy as np
from core.brain import CuriosityEngine, NerveRing


# Action indices
LEFT    = 0
FORWARD = 1
RIGHT   = 2

# Direction vectors [dy, dx] for 4 orientations
# 0=North, 1=East, 2=South, 3=West
DIRECTIONS = [
    (-1,  0),  # North
    ( 0,  1),  # East
    ( 1,  0),  # South
    ( 0, -1),  # West
]


class DataWorm:
    def __init__(self, start_x=None, start_y=None, env_width=40, env_height=40):
        # Body
        self.x = start_x if start_x is not None else env_width  // 2
        self.y = start_y if start_y is not None else env_height // 2
        self.direction = 0  # Facing North initially

        # Brain
        self.curiosity = CuriosityEngine(w_novelty=0.6, w_richness=0.3, w_danger=0.8)
        self.nerve_ring = NerveRing(n_inputs=9, n_hidden=30, n_outputs=3)

        # State tracking
        self.age          = 0       # steps taken
        self.total_reward = 0.0
        self.danger_hits  = 0       # times hit a danger zone
        self.data_found   = 0.0     # cumulative richness consumed
        self.novelty_seen = 0.0     # cumulative novelty

        # Full step log for monitoring
        self.step_log = []

        # Danger reflex threshold — if danger > this, override motor
        self.DANGER_THRESHOLD = 0.6

    def sense(self, env):
        """Read the environment. Returns raw sensor dict."""
        return env.observe(self.x, self.y)

    def _rotate(self, action):
        """Update facing direction."""
        if action == LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == RIGHT:
            self.direction = (self.direction + 1) % 4
        # FORWARD: direction unchanged

    def _move_forward(self, env_width, env_height):
        """Move one step in current direction. Returns new (x, y)."""
        dy, dx = DIRECTIONS[self.direction]
        new_x = np.clip(self.x + dx, 0, env_width  - 1)
        new_y = np.clip(self.y + dy, 0, env_height - 1)
        return int(new_x), int(new_y)

    def step(self, env):
        """
        One complete sense-think-act-learn cycle.

        This is the loop:
        SENSE → CURIOSITY → NERVE RING → REFLEX CHECK → ACT → FEEDBACK → LEARN
        """
        self.age += 1

        # ── 1. SENSE ──────────────────────────────────────
        sensors = self.sense(env)

        # ── 2. CURIOSITY ENGINE ───────────────────────────
        curiosity_out = self.curiosity.compute(sensors)

        # ── 3. NERVE RING (learned integrator) ────────────
        motor_probs = self.nerve_ring.forward(sensors)

        # ── 4. AVOIDANCE REFLEX ───────────────────────────
        # Hardwired. Bypasses everything. If danger is high → retreat.
        reflex_fired = False
        if sensors['danger_here'] > self.DANGER_THRESHOLD:
            # Spin 180 degrees — run
            self.direction = (self.direction + 2) % 4
            action = FORWARD
            reflex_fired = True
            self.danger_hits += 1
        else:
            # ── 5. DECIDE ─────────────────────────────────
            # Blend curiosity bias with learned motor output
            # Curiosity nudges the probabilities, nerve ring finalizes
            direction_bias = curiosity_out['direction_bias']
            blended = motor_probs.copy()
            blended[LEFT]    += max(0, -direction_bias) * 0.3
            blended[RIGHT]   += max(0,  direction_bias) * 0.3
            blended[FORWARD] += curiosity_out['forward_pull'] * 0.2
            blended = np.clip(blended, 0, None)
            blended /= blended.sum() + 1e-8

            # Stochastic action selection (exploration)
            action = int(np.random.choice([LEFT, FORWARD, RIGHT], p=blended))
            self._rotate(action)

        # ── 6. ACT ────────────────────────────────────────
        old_x, old_y = self.x, self.y
        self.x, self.y = self._move_forward(env.width, env.height)

        # ── 7. FEEDBACK from environment ──────────────────
        reward = env.step_into(self.x, self.y)
        self.total_reward += reward
        self.data_found   += max(0, sensors['richness_here'])
        self.novelty_seen += max(0, sensors['novelty_here'])

        # ── 8. LEARN (Hebbian weight update) ──────────────
        weight_change = self.nerve_ring.hebbian_update(reward)

        # ── 9. LOG everything for monitoring ──────────────
        log_entry = {
            'step':           self.age,
            'x':              self.x,
            'y':              self.y,
            'old_x':          old_x,
            'old_y':          old_y,
            'action':         ['LEFT', 'FORWARD', 'RIGHT'][action],
            'direction':      ['N', 'E', 'S', 'W'][self.direction],
            'reflex_fired':   reflex_fired,
            'reward':         round(reward, 4),
            'total_reward':   round(self.total_reward, 4),
            'curiosity_score': round(curiosity_out['curiosity_score'], 4),
            'direction_bias':  round(curiosity_out['direction_bias'], 4),
            'sensors': {k: round(v, 3) for k, v in sensors.items()},
            'motor_probs': [round(float(p), 3) for p in motor_probs],
            'weight_change':  round(weight_change, 6),
            'data_found':     round(self.data_found, 3),
            'novelty_seen':   round(self.novelty_seen, 3),
            'danger_hits':    self.danger_hits,
        }
        self.step_log.append(log_entry)

        return log_entry

    def get_stats(self):
        """Summary statistics for monitoring."""
        if not self.step_log:
            return {}
        recent = self.step_log[-20:]
        return {
            'age':            self.age,
            'position':       (self.x, self.y),
            'total_reward':   round(self.total_reward, 3),
            'avg_reward_recent': round(np.mean([s['reward'] for s in recent]), 4),
            'data_found':     round(self.data_found, 3),
            'novelty_seen':   round(self.novelty_seen, 3),
            'danger_hits':    self.danger_hits,
            'avg_curiosity':  round(np.mean([s['curiosity_score'] for s in recent]), 4),
            'weight_stats':   self.nerve_ring.get_weight_stats(),
            'reflex_rate':    round(sum(1 for s in self.step_log if s['reflex_fired']) / max(1, self.age), 3),
        }
