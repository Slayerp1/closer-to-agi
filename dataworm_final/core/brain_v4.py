"""
brain_v4.py
-----------
THE PREDICTIVE BRAIN.

Adds to brain_v3:
  - PredictiveLayer: mentally simulates outcomes before acting
  - Learns to predict reward for each possible action
  - 1-step lookahead (not tree search, just "if I go left, what happens?")
  - Prediction error drives its own learning

This is the transition from REACTIVE to DELIBERATIVE.
The worm stops just responding and starts ANTICIPATING.

Total brain size: ~50 neurons (nerve ring 30 + predictor 20)
Real C. elegans: 302 neurons. We're still well under budget.
"""

import numpy as np


class PredictiveLayer:
    """
    A small network that predicts: given current sensors + proposed action,
    what reward will I get?

    Input:  sensor_vector (12-18) + action_onehot (3) = 15-21
    Hidden: 20 neurons
    Output: 1 scalar (predicted reward)

    Learning: prediction error (predicted - actual) drives weight update.
    This is supervised learning from the worm's OWN experience.
    No external labels — the environment IS the teacher.
    """

    def __init__(self, n_sensor_inputs=12, n_hidden=20, seed=42):
        self.rng = np.random.default_rng(seed)
        self.n_sensor_inputs = n_sensor_inputs
        self.n_actions = 3
        self.n_inputs = n_sensor_inputs + self.n_actions
        self.n_hidden = n_hidden

        # Weights
        self.W1 = self.rng.normal(0, 0.05, (n_hidden, self.n_inputs))
        self.W2 = self.rng.normal(0, 0.05, (1, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.b2 = np.zeros(1)

        # Learning rate
        self.lr = 0.01

        # Stats
        self.prediction_errors = []
        self.total_predictions = 0

    def _activate(self, z):
        return np.tanh(z)

    def predict(self, sensor_vec, action_idx):
        """
        Predict reward for a specific action given current sensors.
        Returns predicted reward scalar.
        """
        # Build input: sensors + one-hot action
        action_onehot = np.zeros(self.n_actions)
        action_onehot[action_idx] = 1.0

        # Handle sensor vector size mismatch
        if len(sensor_vec) < self.n_sensor_inputs:
            sensor_vec = np.concatenate([sensor_vec,
                np.zeros(self.n_sensor_inputs - len(sensor_vec))])
        elif len(sensor_vec) > self.n_sensor_inputs:
            self._grow(len(sensor_vec))

        x = np.concatenate([sensor_vec, action_onehot])

        h = self._activate(self.W1 @ x + self.b1)
        pred = float((self.W2 @ h + self.b2)[0])

        self.total_predictions += 1
        return pred, x, h

    def predict_all_actions(self, sensor_vec):
        """
        Predict reward for ALL 3 actions. Returns array of [pred_left, pred_forward, pred_right].
        This IS the mental simulation — trying all options before choosing.
        """
        predictions = []
        for action in range(3):
            pred, _, _ = self.predict(sensor_vec, action)
            predictions.append(pred)
        return np.array(predictions)

    def learn(self, sensor_vec, action_idx, actual_reward):
        """
        Update weights based on prediction error.
        predicted - actual = error → adjust weights to reduce error.
        """
        pred, x, h = self.predict(sensor_vec, action_idx)
        error = pred - actual_reward

        self.prediction_errors.append(abs(error))
        if len(self.prediction_errors) > 200:
            self.prediction_errors = self.prediction_errors[-200:]

        # Backprop through the tiny network (only 2 layers, simple)
        # dL/dW2 = error * h
        dW2 = self.lr * error * h.reshape(1, -1)
        db2 = self.lr * error

        # dL/dW1 = error * W2.T * (1-tanh²(z)) * x
        delta_h = error * self.W2.flatten() * (1 - h**2)
        dW1 = self.lr * np.outer(delta_h, x)
        db1 = self.lr * delta_h

        self.W2 -= dW2
        self.b2 -= db2
        self.W1 -= dW1
        self.b1 -= db1

        return float(error)

    def _grow(self, new_sensor_size):
        """Grow input layer for new sensor channels."""
        if new_sensor_size <= self.n_sensor_inputs:
            return
        extra = new_sensor_size - self.n_sensor_inputs
        new_cols = self.rng.normal(0, 0.05, (self.n_hidden, extra))
        # Insert before the action columns
        action_cols = self.W1[:, -self.n_actions:]
        sensor_cols = self.W1[:, :-self.n_actions]
        self.W1 = np.hstack([sensor_cols, new_cols, action_cols])
        self.n_sensor_inputs = new_sensor_size
        self.n_inputs = new_sensor_size + self.n_actions

    def get_stats(self):
        recent_errors = self.prediction_errors[-50:] if self.prediction_errors else [0]
        return {
            'avg_prediction_error': float(np.mean(recent_errors)),
            'total_predictions': self.total_predictions,
            'n_inputs': self.n_inputs,
            'prediction_improving': (
                np.mean(self.prediction_errors[-50:]) < np.mean(self.prediction_errors[:50])
                if len(self.prediction_errors) > 100 else False
            ),
        }


# Import everything from brain_v3
from core.brain_v3 import PlasticCuriosityEngine
from core.brain_v2 import AdaptiveSensorArray, NerveRingV2
