"""
agent_v3.py
-----------
THE PATCHED WORM — 4 biological fixes.

Patch 1: SENSORIMOTOR WALL LOOP
  - Check wall AFTER rotation, before moving
  - Try all directions before giving up
  - Dead-end detection → 180 reverse

Patch 2: GRADED DANGER AVOIDANCE
  - Low danger → soft turn bias (not reflex)
  - Medium danger → strong turn bias
  - High danger → full reversal (emergency reflex)
  - Mirrors real ASH nociceptor graded response

Patch 3: SPATIAL MEMORY
  - Decaying memory of reward at each location
  - Sensed as gradient around current position
  - "The smell of its own past"
  - Memory decays on revisit (habituation)

Patch 4: ADAPTIVE STATE SWITCHING
  - Rate-of-change drives roaming/dwelling transitions
  - Adaptive threshold lowers during scarcity
  - Cooldown prevents oscillation
  - Models serotonin/dopamine modulation
"""

import numpy as np
from core.brain_v2 import CuriosityEngine, NerveRingV2, AdaptiveSensorArray

LEFT    = 0
FORWARD = 1
RIGHT   = 2

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class DataWormV3:
    def __init__(self, start_x=None, start_y=None, env_width=40, env_height=40):
        self.x = start_x if start_x is not None else env_width // 2
        self.y = start_y if start_y is not None else env_height // 2
        self.direction = 0
        self.env_width = env_width
        self.env_height = env_height

        # Brain
        self.curiosity = CuriosityEngine(w_novelty=0.6, w_richness=0.3, w_danger=0.8)
        self.sensors = AdaptiveSensorArray()
        self.nerve_ring = NerveRingV2(n_inputs=12, n_hidden=30, n_outputs=3)

        # ── PATCH 3: SPATIAL MEMORY ──
        # Decaying map of rewards experienced at each location
        # The worm can't see this map — it only senses the LOCAL gradient
        self.memory_map = np.zeros((env_height, env_width))
        self.MEMORY_DECAY = 0.997      # per-step global decay
        self.MEMORY_WRITE_RATE = 0.3   # how much of reward gets written
        self.MEMORY_REVISIT_DECAY = 0.5  # visiting halves the memory (habituation)

        # ── PATCH 4: ADAPTIVE STATE SWITCHING ──
        self.state = 'roaming'
        self.state_cooldown = 0         # minimum steps before next switch
        self.STATE_COOLDOWN_PERIOD = 25 # biological: neuromodulators are slow
        self.reward_baseline = 0.2      # adaptive threshold for "good enough"
        self.reward_ema_fast = 0.0      # fast-moving average (recent)
        self.reward_ema_slow = 0.0      # slow-moving average (trend)

        # Stats & tracking
        self.age = 0
        self.total_reward = 0.0
        self.danger_hits = 0
        self.data_found = 0.0
        self.novelty_seen = 0.0
        self.wall_bumps = 0
        self.unknown_encounters = 0
        self.adaptations = []
        self.recent_rewards = []
        self.position_history = []
        self.step_log = []

    # ─────────────────────────────────────────────
    # PATCH 3: Spatial memory — sense the gradient
    # ─────────────────────────────────────────────

    def _sense_memory(self):
        """
        Read the memory gradient around current position.
        NOT the full map. Just what's nearby. Like a smell.
        """
        def safe_mem(px, py):
            px = np.clip(px, 0, self.env_width - 1)
            py = np.clip(py, 0, self.env_height - 1)
            return float(self.memory_map[py, px])

        return {
            'memory_L': safe_mem(self.x - 1, self.y),
            'memory_R': safe_mem(self.x + 1, self.y),
            'memory_F': safe_mem(self.x, self.y - 1),
            'memory_here': safe_mem(self.x, self.y),
        }

    def _update_memory(self, reward):
        """Write reward to memory map at current location. Decay globally."""
        # Global decay — everything fades
        self.memory_map *= self.MEMORY_DECAY

        # Write current experience
        # Positive rewards get remembered more strongly
        write_val = reward * self.MEMORY_WRITE_RATE
        self.memory_map[self.y, self.x] = (
            self.memory_map[self.y, self.x] * self.MEMORY_REVISIT_DECAY + write_val
        )

    # ─────────────────────────────────────────────
    # PATCH 4: Adaptive state switching
    # ─────────────────────────────────────────────

    def _update_behavioral_state(self, reward):
        """
        Serotonin/dopamine model:
        - reward_ema_fast = "dopamine" (recent events)
        - reward_ema_slow = "serotonin" (baseline state)
        - Rate of change = fast - slow
        - Positive rate → things are getting better → stay
        - Negative rate → things are getting worse → switch
        """
        # Exponential moving averages
        self.reward_ema_fast = 0.1 * reward + 0.9 * self.reward_ema_fast
        self.reward_ema_slow = 0.02 * reward + 0.98 * self.reward_ema_slow

        # Adaptive baseline — lowers during scarcity
        # "The worm lowers its standards when food is scarce"
        self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * self.reward_ema_slow

        # Rate of change
        reward_rate = self.reward_ema_fast - self.reward_ema_slow

        # Cooldown
        if self.state_cooldown > 0:
            self.state_cooldown -= 1
            return

        if self.state == 'roaming':
            # Switch to dwelling if: reward is above adaptive baseline AND improving
            if self.reward_ema_fast > self.reward_baseline and reward_rate > -0.01:
                self.state = 'dwelling'
                self.state_cooldown = self.STATE_COOLDOWN_PERIOD
        else:
            # Switch to roaming if: reward is declining OR below baseline for too long
            if reward_rate < -0.02 or self.reward_ema_fast < self.reward_baseline * 0.5:
                self.state = 'roaming'
                self.state_cooldown = self.STATE_COOLDOWN_PERIOD

    # ─────────────────────────────────────────────
    # PATCH 2: Graded danger avoidance
    # ─────────────────────────────────────────────

    def _compute_danger_bias(self, sensors):
        """
        Graded avoidance — not binary.
        Returns:
          danger_level: 'none', 'low', 'medium', 'emergency'
          turn_bias: float — how strongly to turn away (and which direction)
          should_reverse: bool — emergency full reversal
        """
        d_here = sensors.get('danger_here', 0)
        d_F = sensors.get('danger_F', 0)
        d_L = sensors.get('danger_L', 0)
        d_R = sensors.get('danger_R', 0)

        # Worst danger in the forward cone
        forward_danger = max(d_here, d_F)

        # Emergency reversal
        if forward_danger > 0.7:
            return 'emergency', 0.0, True

        # Which direction is safer?
        # Negative = left is safer, Positive = right is safer
        danger_asym = d_L - d_R  # positive means left is MORE dangerous → go right

        if forward_danger > 0.3:
            # Strong avoidance — heavy bias away
            return 'medium', danger_asym * 2.0, False
        elif forward_danger > 0.1:
            # Soft avoidance — slight bias away
            return 'low', danger_asym * 0.8, False
        else:
            return 'none', 0.0, False

    # ─────────────────────────────────────────────
    # PATCH 1: Sensorimotor wall loop
    # ─────────────────────────────────────────────

    def _try_move(self, env):
        """
        Try to move forward. If wall, try turning.
        Sensorimotor LOOP: sense → try → sense again → try again.
        Returns (new_x, new_y, wall_bump_count)
        """
        bumps = 0

        for attempt in range(4):  # try all 4 directions max
            dy, dx = DIRECTIONS[self.direction]
            new_x = np.clip(self.x + dx, 0, env.width - 1)
            new_y = np.clip(self.y + dy, 0, env.height - 1)

            if not env.is_wall(new_x, new_y):
                return int(new_x), int(new_y), bumps

            # Wall hit — try another direction
            bumps += 1
            if attempt < 2:
                # First two attempts: turn right (systematic search)
                self.direction = (self.direction + 1) % 4
            elif attempt == 2:
                # Third: try left from original
                self.direction = (self.direction + 2) % 4
            else:
                # Completely stuck — stay in place
                return self.x, self.y, bumps

        return self.x, self.y, bumps

    # ─────────────────────────────────────────────
    # Stuck detection
    # ─────────────────────────────────────────────

    def _is_stuck(self):
        if len(self.position_history) < 20:
            return False
        recent = self.position_history[-20:]
        unique = set(recent)
        return len(unique) < 5

    def sense(self, env):
        return env.observe(self.x, self.y)

    def _rotate(self, action):
        if action == LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == RIGHT:
            self.direction = (self.direction + 1) % 4

    # ─────────────────────────────────────────────
    # THE MAIN LOOP
    # ─────────────────────────────────────────────

    def step(self, env):
        """
        SENSE → MEMORY → PROCESS → CURIOSITY → DANGER GRADE →
        NERVE RING → STATE MODULATE → DECIDE → WALL LOOP → ACT → LEARN
        """
        self.age += 1

        # ── 1. SENSE ──
        raw_sensors = self.sense(env)

        # ── 2. SPATIAL MEMORY sensing ──
        memory_sense = self._sense_memory()

        # ── 3. ADAPTIVE SENSOR PROCESSING ──
        sensor_vec, found_unknown = self.sensors.process(raw_sensors)

        if found_unknown and self.unknown_encounters == 0:
            self.adaptations.append({
                'step': self.age,
                'type': 'UNKNOWN_DETECTED',
                'desc': 'First encounter with unknown signal. Growing new neural connections.',
            })
        if found_unknown:
            self.unknown_encounters += 1

        # ── 4. CURIOSITY ──
        curiosity_out = self.curiosity.compute(raw_sensors)

        # ── 5. GRADED DANGER ASSESSMENT (Patch 2) ──
        danger_level, danger_turn_bias, should_reverse = self._compute_danger_bias(raw_sensors)

        # ── 6. NERVE RING ──
        motor_probs = self.nerve_ring.forward(sensor_vec)

        # ── 7. STUCK DETECTION ──
        is_stuck = self._is_stuck()
        if is_stuck and self.age % 5 == 0:
            self.direction = (self.direction + np.random.randint(1, 4)) % 4

        # ── 8. DECIDE ──
        reflex_fired = False

        if should_reverse:
            # EMERGENCY: full reversal (Patch 2 — only for extreme danger)
            self.direction = (self.direction + 2) % 4
            action = FORWARD
            reflex_fired = True
            self.danger_hits += 1

        else:
            # Normal decision with graded influences
            direction_bias = curiosity_out['direction_bias']
            blended = motor_probs.copy()

            # A) Curiosity influence
            if self.state == 'roaming':
                blended[LEFT]    += max(0, -direction_bias) * 0.4
                blended[RIGHT]   += max(0,  direction_bias) * 0.4
                blended[FORWARD] += curiosity_out['forward_pull'] * 0.15
            else:
                blended[FORWARD] += 0.3
                blended[LEFT]    += max(0, -direction_bias) * 0.15
                blended[RIGHT]   += max(0,  direction_bias) * 0.15

            # B) Graded danger avoidance (Patch 2)
            if danger_level == 'medium':
                # Strong turn-away pressure
                if danger_turn_bias > 0:
                    blended[RIGHT] += abs(danger_turn_bias) * 0.5
                    blended[FORWARD] *= 0.3
                else:
                    blended[LEFT] += abs(danger_turn_bias) * 0.5
                    blended[FORWARD] *= 0.3
            elif danger_level == 'low':
                # Gentle bias
                if danger_turn_bias > 0:
                    blended[RIGHT] += abs(danger_turn_bias) * 0.2
                else:
                    blended[LEFT] += abs(danger_turn_bias) * 0.2

            # C) Spatial memory influence (Patch 3)
            # Bias toward remembered good locations when food is scarce
            if self.state == 'roaming' and self.reward_ema_fast < self.reward_baseline:
                mem_asym = memory_sense['memory_R'] - memory_sense['memory_L']
                mem_forward = memory_sense['memory_F']
                blended[RIGHT]   += max(0,  mem_asym) * 0.3
                blended[LEFT]    += max(0, -mem_asym) * 0.3
                blended[FORWARD] += mem_forward * 0.2

            # Normalize
            blended = np.clip(blended, 0.01, None)
            blended /= blended.sum()

            action = int(np.random.choice([LEFT, FORWARD, RIGHT], p=blended))
            self._rotate(action)

        # ── 9. SENSORIMOTOR WALL LOOP (Patch 1) ──
        old_x, old_y = self.x, self.y
        self.x, self.y, wall_bump_count = self._try_move(env)
        self.wall_bumps += wall_bump_count

        # ── 10. FEEDBACK ──
        reward = env.step_into(self.x, self.y)
        if wall_bump_count > 0:
            reward -= 0.1 * wall_bump_count

        self.total_reward += reward
        self.data_found += max(0, raw_sensors.get('richness_here', 0))
        self.novelty_seen += max(0, raw_sensors.get('novelty_here', 0))
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-100:]

        self.position_history.append((self.x, self.y))
        if len(self.position_history) > 50:
            self.position_history = self.position_history[-50:]

        # ── 11. UPDATE SPATIAL MEMORY (Patch 3) ──
        self._update_memory(reward)

        # ── 12. UPDATE BEHAVIORAL STATE (Patch 4) ──
        self._update_behavioral_state(reward)

        # ── 13. LEARN ──
        weight_change, surprise = self.nerve_ring.hebbian_update(reward)

        # ── 14. LOG ──
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
            'danger_level':   danger_level,
            'wall_bumps_step': wall_bump_count,
            'is_stuck':       is_stuck,
            'reward':         round(reward, 4),
            'surprise':       round(surprise, 4),
            'total_reward':   round(self.total_reward, 4),
            'curiosity_score': round(curiosity_out['curiosity_score'], 4),
            'direction_bias':  round(curiosity_out['direction_bias'], 4),
            'reward_ema_fast': round(self.reward_ema_fast, 4),
            'reward_ema_slow': round(self.reward_ema_slow, 4),
            'reward_baseline': round(self.reward_baseline, 4),
            'memory_here':    round(memory_sense['memory_here'], 4),
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
            'reward_baseline': round(self.reward_baseline, 4),
            'weight_stats':   self.nerve_ring.get_weight_stats(),
            'adaptations':    self.adaptations,
            'reflex_rate':    round(sum(1 for s in self.step_log if s['reflex_fired']) / max(1, self.age), 3),
            'n_inputs':       self.nerve_ring.n_inputs,
        }
