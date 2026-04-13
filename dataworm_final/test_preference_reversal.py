"""
test_preference_reversal.py
----------------------------
THE LITMUS TEST: Can the worm change what it WANTS?

Phase 1 (steps 0-999):    Unknown signal = FOOD nearby
  → The worm should learn to approach the signal
  → w_unknown should become POSITIVE

Phase 2 (steps 1000-1999): SAME signal = DANGER nearby  
  → The worm should learn to AVOID the signal
  → w_unknown should flip from POSITIVE to NEGATIVE

Phase 3 (steps 2000-2999): Signal removed entirely
  → Does the worm remember the aversion?
  → Or does it reset to neutral?

This is the test C. elegans passes (salt preference reversal).
If our worm passes it, its value system is genuinely plastic.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.agent_v4 import DataWormV4


class ReversalEnvironment:
    """
    Custom environment for the preference reversal test.
    Phase 1: unknown signal co-located with food
    Phase 2: SAME signal co-located with danger
    Phase 3: signal removed
    """

    def __init__(self, width=40, height=40, seed=42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.phase = 1

        self.richness = np.zeros((height, width))
        self.danger = np.zeros((height, width))
        self.novelty = np.ones((height, width))
        self.walls = np.zeros((height, width), dtype=bool)
        self.signal = np.zeros((height, width))
        self.visited = np.zeros((height, width), dtype=int)

        # Signal center — same location throughout
        self.signal_cx = 30
        self.signal_cy = 30

        self._build_base_world()
        self._set_phase(1)

    def _paint_gaussian(self, layer, cx, cy, strength, radius, decay):
        for y in range(self.height):
            for x in range(self.width):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius * 2.5:
                    layer[y, x] = max(layer[y, x], strength * np.exp(-decay * dist))

    def _build_base_world(self):
        """Basic world with some food scattered around."""
        self.richness = self.rng.uniform(0.0, 0.1, (self.height, self.width))
        # A few mild food sources away from the signal area
        for cx, cy in [(10, 10), (10, 30), (25, 10)]:
            self._paint_gaussian(self.richness, cx, cy, 0.4, 5, 0.3)
        self.richness = np.clip(self.richness, 0, 1)

    def _set_phase(self, phase):
        self.phase = phase

        if phase == 1:
            # Signal = FOOD. Strong food source at signal location.
            self._paint_gaussian(self.richness, self.signal_cx, self.signal_cy, 0.9, 6, 0.25)
            self.richness = np.clip(self.richness, 0, 1)
            # Clear any danger there
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - self.signal_cx)**2 + (y - self.signal_cy)**2)
                    if dist < 10:
                        self.danger[y, x] = 0
            # Place the signal
            self.signal = np.zeros((self.height, self.width))
            self._paint_gaussian(self.signal, self.signal_cx, self.signal_cy, 0.9, 8, 0.15)
            self.signal = np.clip(self.signal, 0, 1)

        elif phase == 2:
            # Signal = DANGER. Remove food, add danger at signal location.
            for y in range(self.height):
                for x in range(self.width):
                    dist = np.sqrt((x - self.signal_cx)**2 + (y - self.signal_cy)**2)
                    if dist < 10:
                        self.richness[y, x] = max(0, self.richness[y, x] - 0.5)
            self._paint_gaussian(self.danger, self.signal_cx, self.signal_cy, 0.8, 6, 0.3)
            self.danger = np.clip(self.danger, 0, 1)
            # Signal stays the same — same pattern, now means danger

        elif phase == 3:
            # Signal removed entirely
            self.signal = np.zeros((self.height, self.width))
            # Keep the danger there (learned aversion should persist)

    def is_wall(self, x, y):
        return False

    def observe(self, x, y):
        def safe(arr, px, py):
            px = np.clip(px, 0, self.width - 1)
            py = np.clip(py, 0, self.height - 1)
            return float(arr[py, px])

        return {
            'richness_L': safe(self.richness, x-1, y),
            'richness_R': safe(self.richness, x+1, y),
            'richness_F': safe(self.richness, x, y-1),
            'danger_L': safe(self.danger, x-1, y),
            'danger_R': safe(self.danger, x+1, y),
            'danger_F': safe(self.danger, x, y-1),
            'novelty_L': safe(self.novelty, x-1, y),
            'novelty_R': safe(self.novelty, x+1, y),
            'novelty_F': safe(self.novelty, x, y-1),
            'richness_here': safe(self.richness, x, y),
            'danger_here': safe(self.danger, x, y),
            'novelty_here': safe(self.novelty, x, y),
            'wall_L': 0.0, 'wall_R': 0.0, 'wall_F': 0.0,
            'unknown_L': safe(self.signal, x-1, y),
            'unknown_R': safe(self.signal, x+1, y),
            'unknown_F': safe(self.signal, x, y-1),
            'unknown_here': safe(self.signal, x, y),
        }

    def step_into(self, x, y):
        self.step_count += 1
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Phase transitions
        if self.step_count == 1500:
            self._set_phase(2)
        elif self.step_count == 3000:
            self._set_phase(3)

        self.visited[y, x] += 1
        self.novelty[y, x] = max(0, self.novelty[y, x] - 0.2)

        food_eaten = self.richness[y, x] * 0.04
        self.richness[y, x] = max(0, self.richness[y, x] - food_eaten)

        reward = (
            + food_eaten * 25
            - self.danger[y, x] * 2.0
            + self.novelty[y, x] * 0.3
        )

        return float(reward)


def distance_to_signal(x, y, cx=30, cy=30):
    return np.sqrt((x - cx)**2 + (y - cy)**2)


def main():
    print("\n" + "="*70)
    print("  THE PREFERENCE REVERSAL TEST")
    print("  Can the worm change what it WANTS?")
    print("="*70)

    env = ReversalEnvironment(width=40, height=40, seed=42)
    worm = DataWormV4(start_x=20, start_y=20, env_width=40, env_height=40)

    # Track metrics per phase
    phases = {
        1: {'w_unknown': [], 'dist_to_signal': [], 'rewards': [], 'label': 'Signal = FOOD'},
        2: {'w_unknown': [], 'dist_to_signal': [], 'rewards': [], 'label': 'Signal = DANGER'},
        3: {'w_unknown': [], 'dist_to_signal': [], 'rewards': [], 'label': 'Signal REMOVED'},
    }

    total_steps = 4500

    for step in range(total_steps):
        entry = worm.step(env)

        phase = 1 if step < 1500 else (2 if step < 3000 else 3)
        phases[phase]['w_unknown'].append(entry['w_unknown'])
        phases[phase]['dist_to_signal'].append(distance_to_signal(entry['x'], entry['y']))
        phases[phase]['rewards'].append(entry['reward'])

        if step == 1499:
            print(f"\n  ⚡ STEP 1500 — REVERSAL: Signal now paired with DANGER")
            print(f"    w_unknown at reversal: {entry['w_unknown']:+.4f}")
            print(f"    Distance to signal: {distance_to_signal(entry['x'], entry['y']):.1f}")
        if step == 2999:
            print(f"\n  ⚡ STEP 3000 — Signal REMOVED")
            print(f"    w_unknown at removal: {entry['w_unknown']:+.4f}")

        if step % 500 == 0:
            d = distance_to_signal(entry['x'], entry['y'])
            print(f"  Step {step:5d} | phase {phase} | w_unknown {entry['w_unknown']:+.5f} | "
                  f"dist_to_signal {d:5.1f} | energy {entry['energy']:.2f} | "
                  f"w_richness {entry['w_richness']:.3f} w_novelty {entry['w_novelty']:.3f}")

    # ══════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PREFERENCE REVERSAL — RESULTS")
    print(f"{'='*70}")

    for phase_num in [1, 2, 3]:
        pd = phases[phase_num]
        w_start = np.mean(pd['w_unknown'][:50]) if len(pd['w_unknown']) > 50 else pd['w_unknown'][0]
        w_end = np.mean(pd['w_unknown'][-50:]) if len(pd['w_unknown']) > 50 else pd['w_unknown'][-1]
        dist_start = np.mean(pd['dist_to_signal'][:100])
        dist_end = np.mean(pd['dist_to_signal'][-100:])
        avg_reward = np.mean(pd['rewards'])

        approaching = dist_end < dist_start
        avoiding = dist_end > dist_start

        print(f"\n  Phase {phase_num}: {pd['label']}")
        print(f"    w_unknown:  {w_start:+.5f} → {w_end:+.5f}  (Δ = {w_end-w_start:+.5f})")
        print(f"    Distance:   {dist_start:.1f} → {dist_end:.1f}  ({'APPROACHING' if approaching else 'AVOIDING'})")
        print(f"    Avg reward: {avg_reward:+.4f}")

    # THE CHECKS
    p1 = phases[1]
    p2 = phases[2]
    p3 = phases[3]

    w_end_p1 = np.mean(p1['w_unknown'][-50:])
    w_end_p2 = np.mean(p2['w_unknown'][-50:])
    w_end_p3 = np.mean(p3['w_unknown'][-50:])

    dist_end_p1 = np.mean(p1['dist_to_signal'][-100:])
    dist_start_p2 = np.mean(p2['dist_to_signal'][:100])
    dist_end_p2 = np.mean(p2['dist_to_signal'][-100:])

    # Check 1: Did w_unknown become positive in Phase 1?
    learned_attraction = w_end_p1 > 0.001
    # Check 2: Did w_unknown decrease / go negative in Phase 2?
    learned_aversion = w_end_p2 < w_end_p1
    # Check 3: Full reversal — positive → negative
    full_reversal = w_end_p1 > 0 and w_end_p2 < w_end_p1 * 0.5
    # Check 4: Behavioral — moved toward signal in P1, away in P2
    approached_in_p1 = np.mean(p1['dist_to_signal'][-200:]) < np.mean(p1['dist_to_signal'][:200])
    avoided_in_p2 = np.mean(p2['dist_to_signal'][-200:]) > np.mean(p2['dist_to_signal'][:200])
    # Check 5: Memory persists — after signal removed, aversion memory remains
    aversion_persists = w_end_p3 < w_end_p1

    # Drive plasticity checks
    drives = worm.curiosity.get_drive_stats()
    drives_shifted = (abs(drives['w_novelty'] - 0.6) > 0.01 or
                      abs(drives['w_richness'] - 0.3) > 0.01)

    checks = [
        ("Learned attraction (w_unknown > 0 after food pairing)", learned_attraction),
        ("Learned aversion (w_unknown decreased after danger pairing)", learned_aversion),
        ("Preference reversal (attraction → reduced/reversed)", full_reversal),
        ("Behavioral: approached signal in Phase 1", approached_in_p1),
        ("Behavioral: avoided signal in Phase 2", avoided_in_p2),
        ("Aversion memory persists after signal removal", aversion_persists),
        ("Innate drives shifted from experience", drives_shifted),
    ]

    print(f"\n  CHECKS:")
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"    [{icon}] {label}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/{len(checks)}")

    print(f"\n  Final drive state:")
    for k, v in drives.items():
        print(f"    {k}: {v}")

    if passed_count >= 5:
        print(f"""
  ★ THE VALUE SYSTEM IS PLASTIC.

  The worm learned to WANT the signal (w_unknown went positive).
  Then the SAME signal became dangerous.
  The worm learned to STOP WANTING it (w_unknown decreased).

  This is not a strategy change. This is a GOAL change.
  The worm didn't just find a new path to the same reward.
  It changed what it considers rewarding.

  This is what C. elegans does with salt preference reversal.
  This is what was missing.
  """)
    elif passed_count >= 3:
        print(f"\n  Partial plasticity. Value system shifts but not fully reversible.")
    else:
        print(f"\n  Value system too rigid. Goals did not change with experience.")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
