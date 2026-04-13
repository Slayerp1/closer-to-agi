"""
brain_v3.py — THE PLASTIC VALUE SYSTEM (clean rebuild)
"""
import numpy as np

class PlasticCuriosityEngine:
    def __init__(self):
        self.w_novelty = 0.6
        self.w_richness = 0.3
        self.w_danger = 0.8
        self.w_unknown = 0.0
        self.DRIVE_LR = 0.0005
        self.DRIVE_DECAY = 0.9999
        self.INNATE_NOVELTY = 0.6
        self.INNATE_RICHNESS = 0.3
        self.INNATE_DANGER = 0.8
        self.ASSOC_LR = 0.005
        self.value_associations = {'unknown': 0.0}
        self.energy = 1.0
        self.ENERGY_DECAY = 0.002
        self.ENERGY_FOOD_RESTORE = 0.15
        self.ENERGY_CRITICAL = 0.2
        self.drive_history = []
        self.novelty_outcomes = []
        self.richness_outcomes = []
        self.unknown_outcomes = []

    def compute(self, sensors):
        richness_asym = sensors.get('richness_R', 0) - sensors.get('richness_L', 0)
        novelty_asym = sensors.get('novelty_R', 0) - sensors.get('novelty_L', 0)
        danger_asym = sensors.get('danger_R', 0) - sensors.get('danger_L', 0)
        ef = self.energy
        eff_nov = self.w_novelty * ef
        eff_rich = self.w_richness * (2.0 - ef)
        eff_dang = self.w_danger * (0.5 + ef * 0.5)
        direction_bias = eff_rich * richness_asym + eff_nov * novelty_asym - eff_dang * danger_asym
        unknown_asym = sensors.get('unknown_R', 0) - sensors.get('unknown_L', 0)
        direction_bias += self.w_unknown * unknown_asym
        curiosity_score = (eff_nov * sensors.get('novelty_here', 0) +
                           eff_rich * sensors.get('richness_here', 0) -
                           eff_dang * sensors.get('danger_here', 0) +
                           self.w_unknown * sensors.get('unknown_here', 0))
        forward_pull = (eff_rich * sensors.get('richness_F', 0) +
                        eff_nov * sensors.get('novelty_F', 0) -
                        eff_dang * sensors.get('danger_F', 0) +
                        self.w_unknown * sensors.get('unknown_F', 0))
        return {'curiosity_score': float(curiosity_score), 'direction_bias': float(direction_bias),
                'forward_pull': float(forward_pull), 'energy': float(self.energy),
                'w_novelty': float(self.w_novelty), 'w_richness': float(self.w_richness),
                'w_danger': float(self.w_danger), 'w_unknown': float(self.w_unknown)}

    def update_energy(self, food_reward):
        self.energy -= self.ENERGY_DECAY
        if food_reward > 0:
            self.energy += food_reward * self.ENERGY_FOOD_RESTORE
        self.energy = np.clip(self.energy, 0.0, 1.0)

    def update_associations(self, sensors, reward):
        unknown_here = sensors.get('unknown_here', 0)
        danger_here = sensors.get('danger_here', 0)
        richness_here = sensors.get('richness_here', 0)
        if unknown_here > 0.01:
            expected = self.w_unknown * unknown_here
            error = reward - expected
            self.value_associations['unknown'] += self.ASSOC_LR * error * unknown_here
            if danger_here > 0.05 and unknown_here > 0.02:
                p = danger_here * unknown_here
                self.w_unknown -= 0.02 * p
                self.value_associations['unknown'] -= 0.05 * p
            if richness_here > 0.2 and unknown_here > 0.02 and danger_here < 0.05:
                p = richness_here * unknown_here
                self.w_unknown += 0.005 * p
            self.unknown_outcomes.append((unknown_here, reward))

    def update_drives(self, sensors, reward, toward_novelty, toward_richness):
        if toward_novelty: self.novelty_outcomes.append(reward)
        if toward_richness: self.richness_outcomes.append(reward)
        if len(self.novelty_outcomes) >= 50:
            self.w_novelty += self.DRIVE_LR * np.mean(self.novelty_outcomes[-50:])
            self.novelty_outcomes = self.novelty_outcomes[-50:]
        if len(self.richness_outcomes) >= 50:
            self.w_richness += self.DRIVE_LR * np.mean(self.richness_outcomes[-50:])
            self.richness_outcomes = self.richness_outcomes[-50:]
        assoc = self.value_associations['unknown']
        self.w_unknown = 0.99 * self.w_unknown + 0.01 * (assoc * 10)
        self.w_unknown = np.clip(self.w_unknown, -1.0, 1.0)
        self.w_novelty = self.DRIVE_DECAY * self.w_novelty + (1 - self.DRIVE_DECAY) * self.INNATE_NOVELTY
        self.w_richness = self.DRIVE_DECAY * self.w_richness + (1 - self.DRIVE_DECAY) * self.INNATE_RICHNESS
        self.w_danger = self.DRIVE_DECAY * self.w_danger + (1 - self.DRIVE_DECAY) * self.INNATE_DANGER
        self.w_novelty = np.clip(self.w_novelty, 0.05, 1.5)
        self.w_richness = np.clip(self.w_richness, 0.05, 1.5)
        self.w_danger = np.clip(self.w_danger, 0.1, 2.0)
        self.drive_history.append({'w_novelty': float(self.w_novelty), 'w_richness': float(self.w_richness),
                                   'w_danger': float(self.w_danger), 'w_unknown': float(self.w_unknown),
                                   'energy': float(self.energy)})

    def get_drive_stats(self):
        return {'w_novelty': round(self.w_novelty, 4), 'w_richness': round(self.w_richness, 4),
                'w_danger': round(self.w_danger, 4), 'w_unknown': round(self.w_unknown, 4),
                'energy': round(self.energy, 4),
                'assoc_unknown': round(self.value_associations['unknown'], 4)}

from core.brain_v2 import AdaptiveSensorArray, NerveRingV2
