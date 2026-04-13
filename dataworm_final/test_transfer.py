"""
test_transfer.py
----------------
THE TRANSFER & META-LEARNING TEST.

Can the worm carry knowledge between worlds?

3 levels tested:
  Level 2: Skill transfer — same signal type, different world
  Level 3: Abstraction transfer — different signal type, different world
  Meta:    Does it learn World C faster than it learned World A?

Method:
  1. Train worm in World A (glow signal → food) for 1500 steps
  2. Clone the trained brain
  3. Drop trained worm into World B (glow signal, different layout)
  4. Drop fresh newborn worm into same World B
  5. Compare: who finds food faster?
  6. Drop trained worm into World C (vibration signal — NEVER seen before)
  7. Drop fresh worm into same World C
  8. Compare: who learns the new signal faster?
"""

import sys, os, copy
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.environment_v3 import TransferEnvironment
from core.agent_v3 import DataWormV3


def clone_brain(source_worm, env_width=40, env_height=40):
    """Deep copy a worm's brain into a new worm body at start position."""
    new_worm = DataWormV3(start_x=2, start_y=2,
                          env_width=env_width, env_height=env_height)
    # Copy nerve ring weights (the learned knowledge)
    new_worm.nerve_ring.W1 = source_worm.nerve_ring.W1.copy()
    new_worm.nerve_ring.W2 = source_worm.nerve_ring.W2.copy()
    new_worm.nerve_ring.b1 = source_worm.nerve_ring.b1.copy()
    new_worm.nerve_ring.b2 = source_worm.nerve_ring.b2.copy()
    new_worm.nerve_ring.n_inputs = source_worm.nerve_ring.n_inputs

    # Copy sensor discoveries (knows about unknown channel)
    new_worm.sensors.discovered_channels = list(source_worm.sensors.discovered_channels)
    new_worm.sensors.unknown_first_seen = source_worm.sensors.unknown_first_seen

    # Copy eligibility traces (must match weight dimensions)
    new_worm.nerve_ring._trace_W1 = source_worm.nerve_ring._trace_W1.copy()
    new_worm.nerve_ring._trace_W2 = source_worm.nerve_ring._trace_W2.copy()

    # Copy neuromodulation state (expected reward baseline)
    new_worm.nerve_ring._expected_reward = source_worm.nerve_ring._expected_reward

    # Reset position, memory, stats — new body, trained brain
    new_worm.memory_map = np.zeros((env_height, env_width))
    new_worm.age = 0
    new_worm.total_reward = 0.0
    new_worm.step_log = []

    return new_worm


def run_worm(worm, env, n_steps, label=""):
    """Run a worm and collect performance metrics."""
    rewards = []
    cumulative = []
    running = 0
    first_good_food_step = None  # first step with reward > 0.5
    unknown_first_used = None

    for step in range(n_steps):
        entry = worm.step(env)
        rewards.append(entry['reward'])
        running += entry['reward']
        cumulative.append(running)

        if first_good_food_step is None and entry['reward'] > 0.5:
            first_good_food_step = step

        if unknown_first_used is None and entry.get('found_unknown', False):
            unknown_first_used = step

    # Metrics
    n = len(rewards)
    return {
        'label': label,
        'total_reward': running,
        'avg_reward': np.mean(rewards),
        'reward_first_500': np.mean(rewards[:500]) if n >= 500 else np.mean(rewards),
        'reward_last_500': np.mean(rewards[-500:]) if n >= 500 else np.mean(rewards),
        'reward_at_250': cumulative[249] if n >= 250 else cumulative[-1],
        'reward_at_500': cumulative[499] if n >= 500 else cumulative[-1],
        'reward_at_1000': cumulative[999] if n >= 1000 else cumulative[-1],
        'first_good_food': first_good_food_step,
        'unknown_first_used': unknown_first_used,
        'danger_hits': worm.danger_hits,
        'n_inputs': worm.nerve_ring.n_inputs,
        'rewards': rewards,
        'cumulative': cumulative,
    }


def main():
    TRAIN_STEPS = 1500
    TEST_STEPS = 1000

    print("\n" + "="*70)
    print("  THE TRANSFER & META-LEARNING TEST")
    print("  Can knowledge survive the jump between worlds?")
    print("="*70)

    # ══════════════════════════════════════════════
    # PHASE 1: TRAINING in World A
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 1 — Training in World A (glow signal → food)")
    print(f"  {TRAIN_STEPS} steps. The worm learns that glowing = food nearby.")
    print(f"{'─'*70}")

    world_a = TransferEnvironment(width=40, height=40, seed=42,
                                   signal_type='glow', signal_on_step=0)
    trainee = DataWormV3(start_x=2, start_y=2, env_width=40, env_height=40)

    train_result = run_worm(trainee, world_a, TRAIN_STEPS, "World A (training)")

    print(f"\n  Training complete:")
    print(f"    Total reward:  {train_result['total_reward']:+.1f}")
    print(f"    Avg reward:    {train_result['avg_reward']:+.4f}")
    print(f"    Brain inputs:  {train_result['n_inputs']} (grew from 12)")
    print(f"    Signal found:  step {train_result['unknown_first_used']}")
    print(f"    First food:    step {train_result['first_good_food']}")

    # Save the trained brain weights for reference
    trained_W1_unknown = trainee.nerve_ring.W1[:, 12:].copy() if trainee.nerve_ring.n_inputs > 12 else None
    print(f"\n    Unknown signal weights learned:")
    if trained_W1_unknown is not None:
        print(f"      Mean: {np.mean(trained_W1_unknown):+.5f}")
        print(f"      Max:  {np.max(trained_W1_unknown):+.5f}")
        print(f"      Positive%: {100*np.mean(trained_W1_unknown > 0):.0f}%")
        print(f"      (Positive bias = learned that unknown signals are GOOD)")

    # ══════════════════════════════════════════════
    # PHASE 2: SKILL TRANSFER — World B (same signal, new layout)
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 2 — Skill Transfer Test: World B")
    print(f"  Same signal type (glow). Completely different world layout.")
    print(f"  Trained worm vs Newborn worm — who adapts faster?")
    print(f"{'─'*70}")

    # Trained worm in World B
    world_b = TransferEnvironment(width=40, height=40, seed=777,
                                   signal_type='glow', signal_on_step=0)
    trained_in_b = clone_brain(trainee)
    trained_b_result = run_worm(trained_in_b, world_b,
                                 TEST_STEPS, "Trained worm → World B")

    # Newborn worm in identical World B
    world_b2 = TransferEnvironment(width=40, height=40, seed=777,
                                    signal_type='glow', signal_on_step=0)
    newborn_b = DataWormV3(start_x=2, start_y=2, env_width=40, env_height=40)
    newborn_b_result = run_worm(newborn_b, world_b2,
                                 TEST_STEPS, "Newborn worm → World B")

    skill_advantage = trained_b_result['reward_at_500'] - newborn_b_result['reward_at_500']
    early_advantage = trained_b_result['reward_at_250'] - newborn_b_result['reward_at_250']

    print(f"\n  {'Metric':<30s} {'Trained':>10s} {'Newborn':>10s} {'Δ':>10s}")
    print(f"  {'─'*62}")
    for metric in ['total_reward', 'avg_reward', 'reward_first_500', 'reward_at_250', 'reward_at_500']:
        t = trained_b_result[metric]
        n = newborn_b_result[metric]
        delta = t - n
        print(f"  {metric:<30s} {t:>+10.2f} {n:>+10.2f} {delta:>+10.2f} {'✓' if delta > 0 else '✗'}")

    fs_t = trained_b_result['first_good_food']
    fs_n = newborn_b_result['first_good_food']
    print(f"  {'first_good_food (step)':<30s} {str(fs_t):>10s} {str(fs_n):>10s} {'faster ✓' if (fs_t or 9999) < (fs_n or 9999) else 'slower'}")

    skill_transfer = trained_b_result['reward_at_500'] > newborn_b_result['reward_at_500']
    early_skill = trained_b_result['reward_at_250'] > newborn_b_result['reward_at_250']
    print(f"\n  SKILL TRANSFER (cumulative @ 500): {'✓ PASSED' if skill_transfer else '✗ FAILED'}")
    print(f"  EARLY ADVANTAGE (@ 250):           {'✓ YES' if early_skill else '✗ NO'}")
    print(f"  (Transfer = faster early learning, not permanent superiority)")

    # ══════════════════════════════════════════════
    # PHASE 3: ABSTRACTION TRANSFER — World C (NEW signal type)
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 3 — Abstraction Transfer Test: World C")
    print(f"  NEW signal type (vibration). The worm has NEVER seen this.")
    print(f"  But it learned that unknown signals CAN predict food.")
    print(f"  Does that meta-knowledge help it learn vibration faster?")
    print(f"{'─'*70}")

    # Trained worm in World C (vibration signal)
    world_c = TransferEnvironment(width=40, height=40, seed=555,
                                   signal_type='vibration', signal_on_step=0)
    trained_in_c = clone_brain(trainee)
    trained_c_result = run_worm(trained_in_c, world_c,
                                 TEST_STEPS, "Trained worm → World C (vibration)")

    # Newborn in identical World C
    world_c2 = TransferEnvironment(width=40, height=40, seed=555,
                                    signal_type='vibration', signal_on_step=0)
    newborn_c = DataWormV3(start_x=2, start_y=2, env_width=40, env_height=40)
    newborn_c_result = run_worm(newborn_c, world_c2,
                                 TEST_STEPS, "Newborn worm → World C (vibration)")

    print(f"\n  {'Metric':<30s} {'Trained':>10s} {'Newborn':>10s} {'Δ':>10s}")
    print(f"  {'─'*62}")
    for metric in ['total_reward', 'avg_reward', 'reward_first_500', 'reward_at_250', 'reward_at_500']:
        t = trained_c_result[metric]
        n = newborn_c_result[metric]
        delta = t - n
        print(f"  {metric:<30s} {t:>+10.2f} {n:>+10.2f} {delta:>+10.2f} {'✓' if delta > 0 else '✗'}")

    us_t = trained_c_result['unknown_first_used']
    us_n = newborn_c_result['unknown_first_used']
    print(f"  {'signal_detected (step)':<30s} {str(us_t):>10s} {str(us_n):>10s} {'faster ✓' if (us_t or 9999) < (us_n or 9999) else ''}")

    abstraction_transfer = trained_c_result['reward_at_500'] > newborn_c_result['reward_at_500']
    early_abstraction = trained_c_result['reward_at_250'] > newborn_c_result['reward_at_250']
    print(f"\n  ABSTRACTION TRANSFER (@ 500): {'✓ PASSED' if abstraction_transfer else '✗ FAILED'}")
    print(f"  EARLY ADVANTAGE (@ 250):      {'✓ YES' if early_abstraction else '✗ NO'}")

    # ══════════════════════════════════════════════
    # PHASE 4: META-LEARNING CHECK
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  PHASE 4 — Meta-Learning Check")
    print(f"  Did the trained worm learn vibration (World C) FASTER")
    print(f"  than it originally learned glow (World A)?")
    print(f"  If yes → it learned HOW to learn new signals.")
    print(f"{'─'*70}")

    # Compare early performance: World A training vs World C with trained brain
    world_a_early = train_result['reward_first_500']
    world_c_early = trained_c_result['reward_first_500']

    print(f"\n  First 500 steps avg reward:")
    print(f"    World A (first time learning any signal): {world_a_early:+.4f}")
    print(f"    World C (new signal, but experienced):    {world_c_early:+.4f}")
    print(f"    Difference:                               {world_c_early - world_a_early:+.4f}")

    meta_learning = world_c_early > world_a_early
    print(f"\n  META-LEARNING: {'✓ PASSED' if meta_learning else '✗ FAILED'}")
    if meta_learning:
        print(f"  The worm learned vibration→food FASTER than it originally")
        print(f"  learned glow→food. It didn't just learn a fact — it learned")
        print(f"  HOW to learn from new signals.")

    # ══════════════════════════════════════════════
    # FINAL SCORECARD
    # ══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TRANSFER TEST — FINAL SCORECARD")
    print(f"{'='*70}")

    checks = [
        ("Level 2: Skill transfer (same signal, new world)", skill_transfer),
        ("Level 3: Abstraction transfer (new signal, new world)", abstraction_transfer),
        ("Meta: Learned new signal faster than first signal", meta_learning),
    ]

    for label, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"  [{icon}] {label}")

    passed = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed}/3")

    if passed == 3:
        print(f"""
  ★ ALL THREE LEVELS OF TRANSFER CONFIRMED.

  This worm:
    1. Learned glow→food in World A
    2. Carried that skill to World B (new layout) — beat the newborn
    3. Encountered vibration (never seen before) in World C — beat the newborn
    4. Learned vibration FASTER than it originally learned glow

  It didn't just memorize. It abstracted.
  It didn't just transfer. It meta-learned.

  The weights for "unknown signal" inputs became a GENERAL
  prior: "new signals might predict food — pay attention."
  This wasn't programmed. It EMERGED from Hebbian learning.
  """)
    elif passed >= 2:
        print(f"\n  Strong transfer. {3-passed} gap(s) remain.")
    elif passed >= 1:
        print(f"\n  Partial transfer. The worm carries some knowledge but not abstractions.")
    else:
        print(f"\n  No transfer detected. The worm starts fresh each time.")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
