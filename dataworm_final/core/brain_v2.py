"""
brain_v2.py
-----------
UPGRADED BRAIN — now with:

1. CURIOSITY ENGINE (same — innate, hardcoded)
2. NERVE RING (upgraded — dynamic input size, can grow new connections)
3. ADAPTIVE SENSORS (NEW — can detect and learn from unknown signals)
4. NEUROMODULATION (NEW — surprise/dopamine scales learning rate)
5. MEMORY TRACE (NEW — eligibility traces for credit assignment)

The big upgrade: the worm can now encounter a COMPLETELY NEW
type of signal and learn to use it, without being reprogrammed.
"""

import numpy as np


class CuriosityEngine:
    """Same as v1 — innate, hardcoded, cannot be changed."""

    def __init__(self, w_novelty=0.6, w_richness=0.3, w_danger=0.8):
        self.w_novelty  = w_novelty
        self.w_richness = w_richness
        self.w_danger   = w_danger

    def compute(self, sensors):
        richness_asym = sensors.get('richness_R', 0) - sensors.get('richness_L', 0)
        novelty_asym  = sensors.get('novelty_R', 0)  - sensors.get('novelty_L', 0)
        danger_asym   = sensors.get('danger_R', 0)   - sensors.get('danger_L', 0)

        forward_richness = sensors.get('richness_F', 0)
        forward_novelty  = sensors.get('novelty_F', 0)
        forward_danger   = sensors.get('danger_F', 0)

        direction_bias = (
            self.w_richness * richness_asym +
            self.w_novelty  * novelty_asym  -
            self.w_danger   * danger_asym
        )

        curiosity_score = (
            self.w_novelty  * sensors.get('novelty_here', 0) +
            self.w_richness * sensors.get('richness_here', 0) -
            self.w_danger   * sensors.get('danger_here', 0)
        )

        forward_pull = (
            self.w_richness * forward_richness +
            self.w_novelty  * forward_novelty  -
            self.w_danger   * forward_danger
        )

        return {
            'curiosity_score': float(curiosity_score),
            'direction_bias':  float(direction_bias),
            'forward_pull':    float(forward_pull),
        }


class AdaptiveSensorArray:
    """
    NEW — The worm's ability to detect and integrate unknown signals.

    How it works:
    - Maintains a list of KNOWN sensor channels (richness, danger, novelty)
    - When an unknown signal appears, it detects the non-zero values
    - Creates a NEW sensor channel on the fly
    - Assigns initial random weights to it in the nerve ring
    - Over time, Hebbian learning figures out if the signal is useful

    This is like a real organism evolving a new receptor.
    Except it happens in one lifetime, through plasticity.
    """

    def __init__(self):
        # Known channels — the worm is BORN knowing these
        self.known_channels = ['richness', 'danger', 'novelty']
        # Discovered channels — learned during lifetime
        self.discovered_channels = []
        # Track when unknown signals first appeared
        self.unknown_first_seen = None
        self.unknown_exposure_count = 0

    def process(self, raw_sensors):
        """
        Take raw sensor dict → produce the input vector for the nerve ring.
        If unknown signals are detected, integrate them.
        """
        # Standard 9 inputs (3 channels × 3 directions)
        base_vector = [
            raw_sensors.get('richness_L', 0), raw_sensors.get('richness_R', 0), raw_sensors.get('richness_F', 0),
            raw_sensors.get('danger_L', 0),   raw_sensors.get('danger_R', 0),   raw_sensors.get('danger_F', 0),
            raw_sensors.get('novelty_L', 0),  raw_sensors.get('novelty_R', 0),  raw_sensors.get('novelty_F', 0),
        ]

        # Wall avoidance inputs (3 more)
        base_vector.extend([
            raw_sensors.get('wall_L', 0),
            raw_sensors.get('wall_R', 0),
            raw_sensors.get('wall_F', 0),
        ])

        # Check for unknown signals
        unknown_L = raw_sensors.get('unknown_L', 0)
        unknown_R = raw_sensors.get('unknown_R', 0)
        unknown_F = raw_sensors.get('unknown_F', 0)
        unknown_vals = [unknown_L, unknown_R, unknown_F]

        has_unknown = any(v > 0.01 for v in unknown_vals)

        if has_unknown:
            self.unknown_exposure_count += 1
            if self.unknown_first_seen is None:
                self.unknown_first_seen = True
                if 'unknown' not in self.discovered_channels:
                    self.discovered_channels.append('unknown')

            # ── DUAL REPRESENTATION ──
            # Raw L/R/F for spatial specificity (helps same-signal transfer)
            # Abstract features for pattern invariance (helps cross-signal transfer)
            # This mirrors real neurobiology: sensory neurons (raw) feed into
            # interneurons (abstract features) — both reach motor neurons.

            # Raw spatial signals
            base_vector.extend(unknown_vals)

            # Abstract features (pattern-invariant)
            max_val = max(unknown_vals)
            presence = min(1.0, max_val * 3.0)
            direction = unknown_R - unknown_L
            gradient = max(unknown_vals) - min(unknown_vals)
            base_vector.extend([presence, direction, gradient])
        else:
            if 'unknown' in self.discovered_channels:
                base_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(base_vector, dtype=float), has_unknown

    def get_input_size(self):
        """Current input size: 12 base + 6 per discovered channel (raw + abstract)."""
        return 12 + 6 * len(self.discovered_channels)


class NerveRingV2:
    """
    UPGRADED nerve ring.

    Changes from v1:
    - Dynamic input size (grows when new sensors are discovered)
    - Neuromodulation: surprise signal scales learning rate
    - Eligibility traces: actions get credit for delayed rewards
    - Synaptic homeostasis: prevents weight explosion
    """

    def __init__(self, n_inputs=12, n_hidden=30, n_outputs=3, seed=42):
        self.rng = np.random.default_rng(seed)

        self.n_inputs  = n_inputs
        self.n_hidden  = n_hidden
        self.n_outputs = n_outputs

        # Weights
        self.W1 = self.rng.normal(0, 0.1, (n_hidden, n_inputs))
        self.W2 = self.rng.normal(0, 0.1, (n_outputs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.b2 = np.zeros(n_outputs)

        # Activations
        self._last_x = None
        self._last_h = None
        self._last_y = None

        # Eligibility traces — fading memory of recent activity
        self._trace_W1 = np.zeros_like(self.W1)
        self._trace_W2 = np.zeros_like(self.W2)
        self._trace_decay = 0.9  # how fast traces fade

        # Neuromodulation state
        self._reward_history = []
        self._expected_reward = 0.0  # running average

        # Stats
        self.weight_change_history = []
        self.growth_events = []

    def _activate(self, z):
        return np.where(z > 0, z, 0.01 * z)

    def _softmax(self, z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def grow_inputs(self, new_n_inputs):
        """
        CRITICAL — grow the network to accept new sensor channels.
        This is neuroplasticity — growing new synaptic connections
        to a receptor that didn't exist before.
        """
        if new_n_inputs <= self.n_inputs:
            return

        extra = new_n_inputs - self.n_inputs
        # New connections initialized with small random weights
        # Slightly POSITIVE bias — encourages exploration of new signal
        new_cols = self.rng.normal(0.02, 0.05, (self.n_hidden, extra))
        self.W1 = np.hstack([self.W1, new_cols])

        # Grow traces too
        self._trace_W1 = np.hstack([
            self._trace_W1,
            np.zeros((self.n_hidden, extra))
        ])

        old_n = self.n_inputs
        self.n_inputs = new_n_inputs

        self.growth_events.append({
            'old_inputs': old_n,
            'new_inputs': new_n_inputs,
            'extra_connections': extra,
        })

    def forward(self, x_vec):
        """Run forward pass. x_vec is already processed by AdaptiveSensorArray."""
        # Handle size mismatch (grow if needed)
        if len(x_vec) > self.n_inputs:
            self.grow_inputs(len(x_vec))
        elif len(x_vec) < self.n_inputs:
            # Pad with zeros
            x_vec = np.concatenate([x_vec, np.zeros(self.n_inputs - len(x_vec))])

        h = self._activate(self.W1 @ x_vec + self.b1)
        y = self._softmax(self.W2 @ h + self.b2)

        self._last_x = x_vec
        self._last_h = h
        self._last_y = y

        # Update eligibility traces — "remember what just fired"
        self._trace_W1 = self._trace_decay * self._trace_W1 + np.outer(h, x_vec)
        self._trace_W2 = self._trace_decay * self._trace_W2 + np.outer(y, h)

        return y

    def hebbian_update(self, reward):
        """
        Upgraded Hebbian with neuromodulation and eligibility traces.

        SURPRISE = |actual_reward - expected_reward|
        High surprise → high learning rate (dopamine analog)
        Low surprise → low learning rate (nothing new to learn)
        """
        if self._last_x is None:
            return 0.0, 0.0

        # ── Neuromodulation: compute surprise ──
        self._reward_history.append(reward)
        if len(self._reward_history) > 50:
            self._reward_history = self._reward_history[-50:]

        self._expected_reward = 0.95 * self._expected_reward + 0.05 * reward
        surprise = abs(reward - self._expected_reward)

        # Surprise scales the learning rate (dopamine!)
        base_lr = 0.005
        modulated_lr = base_lr * (1.0 + 3.0 * surprise)
        modulated_lr = min(modulated_lr, 0.05)  # cap it

        # ── Hebbian update using eligibility traces ──
        # Traces remember RECENT activity, so credit goes to
        # actions from a few steps ago, not just the last one
        reward_signal = reward - self._expected_reward  # prediction error

        dW2 = modulated_lr * reward_signal * self._trace_W2
        dW1 = modulated_lr * reward_signal * self._trace_W1

        # Synaptic homeostasis
        decay = 0.0001
        self.W2 += dW2 - decay * self.W2
        self.W1 += dW1 - decay * self.W1

        change = float(np.abs(dW1).mean() + np.abs(dW2).mean())
        self.weight_change_history.append(change)

        return change, float(surprise)

    def get_weight_stats(self):
        return {
            'W1_mean': float(np.abs(self.W1).mean()),
            'W1_max':  float(np.abs(self.W1).max()),
            'W2_mean': float(np.abs(self.W2).mean()),
            'W2_max':  float(np.abs(self.W2).max()),
            'n_inputs': self.n_inputs,
            'growth_events': len(self.growth_events),
            'recent_change': float(np.mean(self.weight_change_history[-10:])) if self.weight_change_history else 0.0,
        }
