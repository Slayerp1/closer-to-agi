"""
agent_v5.py
-----------
THE DELIBERATIVE WORM.

Before acting, it IMAGINES the outcome of each action.
Blends prediction with curiosity and learned motor output.

Reactive path:  sense → nerve ring → act (fast, reflexive)
Deliberative:   sense → predict each action → pick best → act (slower, smarter)
Actual blend:   both run in parallel, weighted by prediction confidence

Early in life: predictions are bad → mostly reactive (like a baby worm)
Later in life: predictions improve → increasingly deliberative (experienced worm)

This mirrors real neurodevelopment: reflexes come first, planning comes later.
"""

import numpy as np
from core.brain_v4 import PlasticCuriosityEngine, AdaptiveSensorArray, NerveRingV2, PredictiveLayer

LEFT    = 0
FORWARD = 1
RIGHT   = 2
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class DataWormV5:
    def __init__(self, start_x=None, start_y=None, env_width=40, env_height=40):
        self.x = start_x if start_x is not None else env_width // 2
        self.y = start_y if start_y is not None else env_height // 2
        self.direction = 0
        self.env_width = env_width
        self.env_height = env_height

        # Brain components
        self.curiosity = PlasticCuriosityEngine()
        self.sensors = AdaptiveSensorArray()
        self.nerve_ring = NerveRingV2(n_inputs=12, n_hidden=30, n_outputs=3)
        self.predictor = PredictiveLayer(n_sensor_inputs=12, n_hidden=20)

        # Prediction confidence — starts low, grows as predictions improve
        self.prediction_weight = 0.0  # how much to trust predictions (0-1)
        self.PREDICTION_WEIGHT_LR = 0.001  # slow ramp-up

        # Memory + state (from v3/v4)
        self.memory_map = np.zeros((env_height, env_width))
        self.MEMORY_DECAY = 0.997
        self.MEMORY_WRITE_RATE = 0.3
        self.MEMORY_REVISIT_DECAY = 0.5

        self.state = 'roaming'
        self.state_cooldown = 0
        self.STATE_COOLDOWN_PERIOD = 25
        self.reward_baseline = 0.2
        self.reward_ema_fast = 0.0
        self.reward_ema_slow = 0.0

        # Stats
        self.age = 0
        self.total_reward = 0.0
        self.danger_hits = 0
        self.data_found = 0.0
        self.novelty_seen = 0.0
        self.wall_bumps = 0
        self.unknown_encounters = 0
        self.prediction_overrides = 0  # times prediction changed the decision
        self.adaptations = []
        self.recent_rewards = []
        self.position_history = []
        self.step_log = []

    # ── Memory, state, danger, wall methods (same as v4/v3) ──

    def _sense_memory(self):
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
        self.memory_map *= self.MEMORY_DECAY
        self.memory_map[self.y, self.x] = (
            self.memory_map[self.y, self.x] * self.MEMORY_REVISIT_DECAY +
            reward * self.MEMORY_WRITE_RATE
        )

    def _update_behavioral_state(self, reward):
        self.reward_ema_fast = 0.1 * reward + 0.9 * self.reward_ema_fast
        self.reward_ema_slow = 0.02 * reward + 0.98 * self.reward_ema_slow
        self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * self.reward_ema_slow
        if self.curiosity.energy < self.curiosity.ENERGY_CRITICAL:
            self.state = 'roaming'
            return
        reward_rate = self.reward_ema_fast - self.reward_ema_slow
        if self.state_cooldown > 0:
            self.state_cooldown -= 1
            return
        if self.state == 'roaming':
            if self.reward_ema_fast > self.reward_baseline and reward_rate > -0.01:
                self.state = 'dwelling'
                self.state_cooldown = self.STATE_COOLDOWN_PERIOD
        else:
            if reward_rate < -0.02 or self.reward_ema_fast < self.reward_baseline * 0.5:
                self.state = 'roaming'
                self.state_cooldown = self.STATE_COOLDOWN_PERIOD

    def _compute_danger_bias(self, sensors):
        d_here = sensors.get('danger_here', 0)
        d_F = sensors.get('danger_F', 0)
        d_L = sensors.get('danger_L', 0)
        d_R = sensors.get('danger_R', 0)
        forward_danger = max(d_here, d_F)
        threshold = 0.7 if self.curiosity.energy > self.curiosity.ENERGY_CRITICAL else 0.9
        if forward_danger > threshold:
            return 'emergency', 0.0, True
        danger_asym = d_L - d_R
        if forward_danger > 0.3:
            return 'medium', danger_asym * 2.0, False
        elif forward_danger > 0.1:
            return 'low', danger_asym * 0.8, False
        return 'none', 0.0, False

    def _try_move(self, env):
        bumps = 0
        for attempt in range(4):
            dy, dx = DIRECTIONS[self.direction]
            new_x = np.clip(self.x + dx, 0, env.width - 1)
            new_y = np.clip(self.y + dy, 0, env.height - 1)
            if not env.is_wall(new_x, new_y):
                return int(new_x), int(new_y), bumps
            bumps += 1
            if attempt < 2:
                self.direction = (self.direction + 1) % 4
            elif attempt == 2:
                self.direction = (self.direction + 2) % 4
            else:
                return self.x, self.y, bumps
        return self.x, self.y, bumps

    def _is_stuck(self):
        if len(self.position_history) < 20:
            return False
        return len(set(self.position_history[-20:])) < 5

    def sense(self, env):
        return env.observe(self.x, self.y)

    def _rotate(self, action):
        if action == LEFT:
            self.direction = (self.direction - 1) % 4
        elif action == RIGHT:
            self.direction = (self.direction + 1) % 4

    # ── THE MAIN LOOP ──

    def step(self, env):
        self.age += 1

        # ── SENSE ──
        raw_sensors = self.sense(env)
        memory_sense = self._sense_memory()
        sensor_vec, found_unknown = self.sensors.process(raw_sensors)

        if found_unknown and self.unknown_encounters == 0:
            self.adaptations.append({
                'step': self.age, 'type': 'UNKNOWN_DETECTED',
                'desc': 'First unknown signal. Growing connections.',
            })
        if found_unknown:
            self.unknown_encounters += 1

        # ── CURIOSITY (plastic) ──
        curiosity_out = self.curiosity.compute(raw_sensors)

        # ── DANGER ──
        danger_level, danger_turn_bias, should_reverse = self._compute_danger_bias(raw_sensors)

        # ── NERVE RING (reactive) ──
        motor_probs = self.nerve_ring.forward(sensor_vec)

        # ── PREDICTION (deliberative) ──
        # Mentally simulate all 3 actions
        predicted_rewards = self.predictor.predict_all_actions(sensor_vec)

        # Convert predictions to soft preferences (softmax)
        pred_temp = 2.0  # temperature — higher = more exploratory
        pred_exp = np.exp((predicted_rewards - predicted_rewards.max()) / pred_temp)
        pred_probs = pred_exp / pred_exp.sum()

        # ── STUCK ──
        if self._is_stuck() and self.age % 5 == 0:
            self.direction = (self.direction + np.random.randint(1, 4)) % 4

        # ── DECIDE ──
        reflex_fired = False
        prediction_used = False

        if should_reverse:
            self.direction = (self.direction + 2) % 4
            action = FORWARD
            reflex_fired = True
            self.danger_hits += 1
        else:
            direction_bias = curiosity_out['direction_bias']

            # REACTIVE component (nerve ring + curiosity)
            reactive = motor_probs.copy()
            if self.state == 'roaming':
                reactive[LEFT]    += max(0, -direction_bias) * 0.4
                reactive[RIGHT]   += max(0,  direction_bias) * 0.4
                reactive[FORWARD] += curiosity_out['forward_pull'] * 0.15
            else:
                reactive[FORWARD] += 0.3
                reactive[LEFT]    += max(0, -direction_bias) * 0.15
                reactive[RIGHT]   += max(0,  direction_bias) * 0.15

            # Graded danger
            if danger_level == 'medium':
                if danger_turn_bias > 0:
                    reactive[RIGHT] += abs(danger_turn_bias) * 0.5
                    reactive[FORWARD] *= 0.3
                else:
                    reactive[LEFT] += abs(danger_turn_bias) * 0.5
                    reactive[FORWARD] *= 0.3
            elif danger_level == 'low':
                if danger_turn_bias > 0:
                    reactive[RIGHT] += abs(danger_turn_bias) * 0.2
                else:
                    reactive[LEFT] += abs(danger_turn_bias) * 0.2

            # Memory
            if self.state == 'roaming' and self.reward_ema_fast < self.reward_baseline:
                mem_asym = memory_sense['memory_R'] - memory_sense['memory_L']
                reactive[RIGHT]   += max(0,  mem_asym) * 0.3
                reactive[LEFT]    += max(0, -mem_asym) * 0.3
                reactive[FORWARD] += memory_sense['memory_F'] * 0.2

            reactive = np.clip(reactive, 0.01, None)
            reactive /= reactive.sum()

            # ── BLEND reactive + deliberative ──
            # prediction_weight controls the mix (grows with experience)
            pw = self.prediction_weight
            blended = (1 - pw) * reactive + pw * pred_probs

            blended = np.clip(blended, 0.01, None)
            blended /= blended.sum()

            # Check if prediction changed the decision
            reactive_choice = np.argmax(reactive)
            blended_choice = np.argmax(blended)
            if reactive_choice != blended_choice and pw > 0.1:
                prediction_used = True
                self.prediction_overrides += 1

            action = int(np.random.choice([LEFT, FORWARD, RIGHT], p=blended))
            self._rotate(action)

        # ── MOVE ──
        old_x, old_y = self.x, self.y
        self.x, self.y, wall_bump_count = self._try_move(env)
        self.wall_bumps += wall_bump_count

        # ── FEEDBACK ──
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

        # ── UPDATE MEMORY ──
        self._update_memory(reward)

        # ── UPDATE ENERGY ──
        self.curiosity.update_energy(max(0, raw_sensors.get('richness_here', 0)))

        # ── UPDATE ASSOCIATIONS ──
        self.curiosity.update_associations(raw_sensors, reward)

        # ── UPDATE DRIVES ──
        nov_F = raw_sensors.get('novelty_F', 0)
        rich_F = raw_sensors.get('richness_F', 0)
        self.curiosity.update_drives(raw_sensors, reward, nov_F > 0.5, rich_F > 0.2)

        # ── UPDATE STATE ──
        self._update_behavioral_state(reward)

        # ── LEARN: Hebbian (nerve ring) ──
        weight_change, surprise = self.nerve_ring.hebbian_update(reward)

        # ── LEARN: Prediction error (predictor) ──
        pred_error = self.predictor.learn(sensor_vec, action, reward)

        # ── UPDATE PREDICTION CONFIDENCE ──
        # If predictions are getting better, trust them more
        recent_errors = self.predictor.prediction_errors[-50:]
        if len(recent_errors) >= 50:
            avg_error = np.mean(recent_errors)
            # Low error → high confidence
            target_weight = max(0, min(0.5, 1.0 - avg_error * 2))
            self.prediction_weight += self.PREDICTION_WEIGHT_LR * (target_weight - self.prediction_weight)
            self.prediction_weight = np.clip(self.prediction_weight, 0, 0.5)

        # ── LOG ──
        drive_stats = self.curiosity.get_drive_stats()
        pred_stats = self.predictor.get_stats()
        log_entry = {
            'step': self.age,
            'x': self.x, 'y': self.y,
            'action': ['LEFT', 'FORWARD', 'RIGHT'][action],
            'state': self.state,
            'reflex_fired': reflex_fired,
            'prediction_used': prediction_used,
            'prediction_weight': round(self.prediction_weight, 4),
            'predicted_rewards': [round(p, 3) for p in predicted_rewards],
            'pred_error': round(pred_error, 4),
            'avg_pred_error': round(pred_stats['avg_prediction_error'], 4),
            'reward': round(reward, 4),
            'total_reward': round(self.total_reward, 4),
            'energy': drive_stats['energy'],
            'w_unknown': drive_stats['w_unknown'],
            'w_novelty': drive_stats['w_novelty'],
            'w_richness': drive_stats['w_richness'],
            'weight_change': round(weight_change, 6),
            'n_inputs': self.nerve_ring.n_inputs,
            'found_unknown': found_unknown,
            'data_found': round(self.data_found, 3),
            'danger_hits': self.danger_hits,
            'wall_bumps': self.wall_bumps,
            'prediction_overrides': self.prediction_overrides,
        }
        self.step_log.append(log_entry)
        return log_entry

    def get_stats(self):
        if not self.step_log:
            return {}
        recent = self.step_log[-20:]
        return {
            'age': self.age,
            'total_reward': round(self.total_reward, 3),
            'drives': self.curiosity.get_drive_stats(),
            'prediction': self.predictor.get_stats(),
            'prediction_weight': round(self.prediction_weight, 4),
            'prediction_overrides': self.prediction_overrides,
            'n_inputs': self.nerve_ring.n_inputs,
            'danger_hits': self.danger_hits,
            'wall_bumps': self.wall_bumps,
        }
