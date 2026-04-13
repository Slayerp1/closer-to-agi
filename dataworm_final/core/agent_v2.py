"""
agent_v2.py
-----------
THE UPGRADED WORM.

New capabilities:
  - Adaptive sensors (can detect unknown signal types)
  - Wall awareness (can't walk through walls)
  - Neuromodulated learning (surprise drives plasticity)
  - Behavioral states (roaming vs dwelling — like real C. elegans)
  - Stuck detection (knows when it's going in circles)
"""

import numpy as np
from core.brain_v2 import CuriosityEngine, NerveRingV2, AdaptiveSensorArray

LEFT    = 0
FORWARD = 1
RIGHT   = 2

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class DataWormV2:
    def __init__(self, start_x=None, start_y=None, env_width=40, env_height=40):
        self.x = start_x if start_x is not None else env_width // 2
        self.y = start_y if start_y is not None else env_height // 2
        self.direction = 0

        # Brain components
        self.curiosity = CuriosityEngine(w_novelty=0.6, w_richness=0.3, w_danger=0.8)
        self.sensors = AdaptiveSensorArray()
        self.nerve_ring = NerveRingV2(n_inputs=12, n_hidden=30, n_outputs=3)

        # Behavioral state — roaming vs dwelling
        # Real C. elegans switches between these based on food availability
        self.state = 'roaming'  # 'roaming' or 'dwelling'
        self.state_timer = 0
        self.recent_rewards = []

        # Position history for stuck detection
        self.position_history = []

        # Stats
        self.age = 0
        self.total_reward = 0.0
        self.danger_hits = 0
        self.data_found = 0.0
        self.novelty_seen = 0.0
        self.wall_bumps = 0
        self.unknown_encounters = 0
        self.adaptations = []  # log of when it adapted to new things

        self.DANGER_THRESHOLD = 0.5
        self.step_log = []

    def _update_behavioral_state(self):
        """
        Switch between roaming and dwelling.
        Roaming: explore, move fast, seek novelty
        Dwelling: stay near food, move slowly, exploit
        """
        if len(self.recent_rewards) < 10:
            return

        avg_recent = np.mean(self.recent_rewards[-10:])
        avg_older = np.mean(self.recent_rewards[-30:-10]) if len(self.recent_rewards) > 30 else 0

        if self.state == 'roaming':
            # Switch to dwelling if finding good food consistently
            if avg_recent > 0.4:
                self.state = 'dwelling'
                self.state_timer = 0
        else:
            # Switch back to roaming if rewards declining or stayed too long
            self.state_timer += 1
            if avg_recent < 0.2 or self.state_timer > 50:
                self.state = 'roaming'
                self.state_timer = 0

    def _is_stuck(self):
        """Detect if worm is going in circles."""
        if len(self.position_history) < 20:
            return False
        recent = self.position_history[-20:]
        unique = set(recent)
        return len(unique) < 5  # visiting fewer than 5 unique cells in 20 steps

    def sense(self, env):
        return env.observe(self.x, self.y)

    def _rotate(self, action):
        if action == LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == RIGHT:
            self.direction = (self.direction + 1) % 4

    def _move_forward(self, env):
        dy, dx = DIRECTIONS[self.direction]
        new_x = np.clip(self.x + dx, 0, env.width - 1)
        new_y = np.clip(self.y + dy, 0, env.height - 1)

        # WALL CHECK — can't walk through walls
        if env.is_wall(new_x, new_y):
            self.wall_bumps += 1
            return self.x, self.y, True  # bumped

        return int(new_x), int(new_y), False

    def step(self, env):
        """
        SENSE → PROCESS → CURIOSITY → NERVE RING → STATE → REFLEX → ACT → LEARN

        The full loop, now with adaptive sensing and behavioral states.
        """
        self.age += 1

        # ── 1. SENSE raw environment ──
        raw_sensors = self.sense(env)

        # ── 2. ADAPTIVE SENSOR PROCESSING ──
        # This is where unknown signals get detected and integrated
        sensor_vec, found_unknown = self.sensors.process(raw_sensors)

        if found_unknown and self.unknown_encounters == 0:
            self.adaptations.append({
                'step': self.age,
                'type': 'UNKNOWN_DETECTED',
                'desc': 'First encounter with unknown signal. Growing new neural connections.',
            })
        if found_unknown:
            self.unknown_encounters += 1

        # ── 3. CURIOSITY ──
        curiosity_out = self.curiosity.compute(raw_sensors)

        # ── 4. NERVE RING ──
        motor_probs = self.nerve_ring.forward(sensor_vec)

        # ── 5. BEHAVIORAL STATE ──
        self._update_behavioral_state()

        # ── 6. STUCK DETECTION ──
        is_stuck = self._is_stuck()
        if is_stuck and self.age % 5 == 0:
            # Random turn to escape loops
            self.direction = (self.direction + self.nerve_ring.rng.integers(1, 4)) % 4

        # ── 7. AVOIDANCE REFLEX ──
        reflex_fired = False
        wall_bump = False

        # Danger reflex
        if raw_sensors.get('danger_here', 0) > self.DANGER_THRESHOLD:
            self.direction = (self.direction + 2) % 4
            action = FORWARD
            reflex_fired = True
            self.danger_hits += 1

        # Wall reflex — if wall ahead, don't go forward
        elif raw_sensors.get('wall_F', 0) > 0.5:
            # Turn toward less-walled direction
            if raw_sensors.get('wall_L', 0) < raw_sensors.get('wall_R', 0):
                action = LEFT
            else:
                action = RIGHT
            self._rotate(action)
            reflex_fired = True

        else:
            # ── 8. DECIDE ──
            direction_bias = curiosity_out['direction_bias']
            blended = motor_probs.copy()

            # Behavioral state modifies the blend
            if self.state == 'roaming':
                # More exploration — boost novelty-driven turns
                blended[LEFT]    += max(0, -direction_bias) * 0.4
                blended[RIGHT]   += max(0,  direction_bias) * 0.4
                blended[FORWARD] += curiosity_out['forward_pull'] * 0.15
            else:
                # Dwelling — less turning, more forward exploitation
                blended[FORWARD] += 0.3
                blended[LEFT]    += max(0, -direction_bias) * 0.15
                blended[RIGHT]   += max(0,  direction_bias) * 0.15

            blended = np.clip(blended, 0.01, None)
            blended /= blended.sum()

            action = int(np.random.choice([LEFT, FORWARD, RIGHT], p=blended))
            self._rotate(action)

        # ── 9. ACT ──
        old_x, old_y = self.x, self.y
        self.x, self.y, wall_bump = self._move_forward(env)

        # ── 10. FEEDBACK ──
        reward = env.step_into(self.x, self.y)

        # Wall bump penalty
        if wall_bump:
            reward -= 0.3

        self.total_reward += reward
        self.data_found += max(0, raw_sensors.get('richness_here', 0))
        self.novelty_seen += max(0, raw_sensors.get('novelty_here', 0))
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-100:]

        self.position_history.append((self.x, self.y))
        if len(self.position_history) > 50:
            self.position_history = self.position_history[-50:]

        # ── 11. LEARN ──
        weight_change, surprise = self.nerve_ring.hebbian_update(reward)

        # ── 12. LOG ──
        log_entry = {
            'step':           self.age,
            'x':              self.x,
            'y':              self.y,
            'old_x':          old_x,
            'old_y':          old_y,
            'action':         ['LEFT', 'FORWARD', 'RIGHT'][action],
            'direction':      ['N', 'E', 'S', 'W'][self.direction],
            'state':          self.state,
            'reflex_fired':   reflex_fired,
            'wall_bump':      wall_bump,
            'is_stuck':       is_stuck,
            'reward':         round(reward, 4),
            'surprise':       round(surprise, 4),
            'total_reward':   round(self.total_reward, 4),
            'curiosity_score': round(curiosity_out['curiosity_score'], 4),
            'direction_bias':  round(curiosity_out['direction_bias'], 4),
            'sensors': {k: round(v, 3) for k, v in raw_sensors.items()},
            'motor_probs': [round(float(p), 3) for p in motor_probs],
            'weight_change':  round(weight_change, 6),
            'data_found':     round(self.data_found, 3),
            'novelty_seen':   round(self.novelty_seen, 3),
            'danger_hits':    self.danger_hits,
            'wall_bumps':     self.wall_bumps,
            'unknown_encounters': self.unknown_encounters,
            'found_unknown':  found_unknown,
            'n_inputs':       self.nerve_ring.n_inputs,
        }
        self.step_log.append(log_entry)
        return log_entry

    def get_stats(self):
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
            'wall_bumps':     self.wall_bumps,
            'unknown_encounters': self.unknown_encounters,
            'behavioral_state': self.state,
            'avg_curiosity':  round(np.mean([s['curiosity_score'] for s in recent]), 4),
            'avg_surprise':   round(np.mean([s['surprise'] for s in recent]), 4),
            'weight_stats':   self.nerve_ring.get_weight_stats(),
            'adaptations':    self.adaptations,
            'reflex_rate':    round(sum(1 for s in self.step_log if s['reflex_fired']) / max(1, self.age), 3),
            'n_inputs':       self.nerve_ring.n_inputs,
        }
