"""
brain.py
--------
This is the entire intelligence of the system.

Two components:

1. CURIOSITY ENGINE (hardcoded, innate)
   The one drive that cannot be trained away.
   It always pulls the agent toward the most informative unknown.
   This is chemotaxis. This is the seed.

2. NERVE RING (learned, plastic)
   A small spiking-inspired neural network.
   Takes sensor signals → produces motor command.
   Weights updated by Hebbian rule on every step.
   No backpropagation. No gradient descent. No dataset.
   It learns by living.
"""

import numpy as np


# ─────────────────────────────────────────────
# CURIOSITY ENGINE
# The soul. Never changes. Always runs.
# ─────────────────────────────────────────────

class CuriosityEngine:
    """
    Computes an intrinsic drive signal from sensor readings.

    The agent is ALWAYS pulled toward:
      - High novelty (unexplored data regions)
      - High richness (dense, structured data)
      - Away from danger (corrupted/noisy data)

    This is not trained. This is not a reward function given by a human.
    This is the worm's nature. It cannot not be curious.
    """

    def __init__(self, w_novelty=0.6, w_richness=0.3, w_danger=0.8):
        # Weights define what the agent "cares about"
        # novelty > richness because exploration > exploitation at start
        # danger has highest weight because survival > curiosity
        self.w_novelty  = w_novelty
        self.w_richness = w_richness
        self.w_danger   = w_danger

    def compute(self, sensors):
        """
        Returns:
          curiosity_score: float  — overall drive intensity (how curious is it right now)
          direction_bias:  float  — negative=left, positive=right, 0=forward
        """
        # Left/right asymmetry — this is how chemotaxis works
        # If more food/novelty on the right → bias right
        richness_asym = sensors['richness_R'] - sensors['richness_L']
        novelty_asym  = sensors['novelty_R']  - sensors['novelty_L']
        danger_asym   = sensors['danger_R']   - sensors['danger_L']

        # Forward pull — is it better to go straight?
        forward_richness = sensors['richness_F']
        forward_novelty  = sensors['novelty_F']
        forward_danger   = sensors['danger_F']

        # Direction bias: positive = turn right, negative = turn left
        direction_bias = (
            self.w_richness * richness_asym +
            self.w_novelty  * novelty_asym  -
            self.w_danger   * danger_asym
        )

        # Curiosity score: how strong is the drive right now?
        curiosity_score = (
            self.w_novelty  * sensors['novelty_here'] +
            self.w_richness * sensors['richness_here'] -
            self.w_danger   * sensors['danger_here']
        )

        # Forward attractiveness
        forward_pull = (
            self.w_richness * forward_richness +
            self.w_novelty  * forward_novelty  -
            self.w_danger   * forward_danger
        )

        return {
            'curiosity_score': float(curiosity_score),
            'direction_bias':  float(direction_bias),
            'forward_pull':    float(forward_pull),
            'raw': {
                'richness_asym': richness_asym,
                'novelty_asym':  novelty_asym,
                'danger_asym':   danger_asym,
            }
        }


# ─────────────────────────────────────────────
# NERVE RING
# The learned integrator. Plastic. Alive.
# ─────────────────────────────────────────────

class NerveRing:
    """
    A small neural integrator — the worm's brain.

    Input:  9 sensor values (richness L/R/F, danger L/R/F, novelty L/R/F)
    Hidden: n_hidden neurons (default 30, matching C. elegans interneuron count)
    Output: 3 motor commands [turn_left, go_forward, turn_right]

    Learning: Hebbian — "neurons that fire together, wire together"
    No backprop. No dataset. No labels.
    The weights change every time the agent takes a step and gets a reward.
    """

    def __init__(self, n_inputs=9, n_hidden=30, n_outputs=3, seed=42):
        rng = np.random.default_rng(seed)

        # Initialize weights small and random — like a newborn nervous system
        self.W1 = rng.normal(0, 0.1, (n_hidden, n_inputs))   # input → hidden
        self.W2 = rng.normal(0, 0.1, (n_outputs, n_hidden))  # hidden → output
        self.b1 = np.zeros(n_hidden)
        self.b2 = np.zeros(n_outputs)

        self.n_inputs  = n_inputs
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs

        # Store last activations for Hebbian update
        self._last_x  = None
        self._last_h  = None
        self._last_y  = None

        # Track weight history for monitoring
        self.weight_change_history = []

    def _activate(self, z):
        """Leaky ReLU — simple, biologically plausible."""
        return np.where(z > 0, z, 0.01 * z)

    def _softmax(self, z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def forward(self, sensor_dict):
        """
        Run sensors through the network.
        Returns motor probabilities [p_left, p_forward, p_right].
        """
        x = np.array([
            sensor_dict['richness_L'], sensor_dict['richness_R'], sensor_dict['richness_F'],
            sensor_dict['danger_L'],   sensor_dict['danger_R'],   sensor_dict['danger_F'],
            sensor_dict['novelty_L'],  sensor_dict['novelty_R'],  sensor_dict['novelty_F'],
        ], dtype=float)

        h = self._activate(self.W1 @ x + self.b1)
        y = self._softmax(self.W2 @ h + self.b2)

        self._last_x = x
        self._last_h = h
        self._last_y = y

        return y  # [p_left, p_forward, p_right]

    def hebbian_update(self, reward, learning_rate=0.005):
        """
        Hebbian learning rule: strengthen connections that led to reward.
        Weaken connections that led to punishment.

        This is the ONLY learning that happens. No gradient. No loss function.
        The reward is: richness - danger + novelty bonus.
        """
        if self._last_x is None:
            return 0.0

        # dW proportional to pre * post * reward
        # If reward > 0: strengthen the path
        # If reward < 0: weaken the path (anti-Hebbian)
        dW2 = learning_rate * reward * np.outer(self._last_y, self._last_h)
        dW1 = learning_rate * reward * np.outer(self._last_h, self._last_x)

        # Weight decay — prevents runaway growth (like synaptic homeostasis)
        decay = 0.0001
        self.W2 += dW2 - decay * self.W2
        self.W1 += dW1 - decay * self.W1

        # Track total weight change magnitude
        change = float(np.abs(dW1).mean() + np.abs(dW2).mean())
        self.weight_change_history.append(change)

        return change

    def get_weight_stats(self):
        return {
            'W1_mean': float(np.abs(self.W1).mean()),
            'W1_max':  float(np.abs(self.W1).max()),
            'W2_mean': float(np.abs(self.W2).mean()),
            'W2_max':  float(np.abs(self.W2).max()),
            'recent_change': float(np.mean(self.weight_change_history[-10:])) if self.weight_change_history else 0.0,
        }
