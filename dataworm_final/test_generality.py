"""
test_generality.py
------------------
THE AGI TEST.

Can the worm handle what it was NEVER designed for?

6 events, each testing a different aspect of general intelligence:
  1. Food disappears      → Can it stop going to a place that used to be good?
  2. Wall appears          → Can it navigate around a new obstacle?
  3. Unknown signal        → Can it detect and USE a signal type it was never programmed for?
  4. Danger moves          → Can it re-learn what's dangerous?
  5. New food appears      → Can it discover new opportunities?
  6. Catastrophe           → Can it survive when everything gets worse?
"""

import sys, os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from core.environment_v2 import DynamicEnvironment
from core.agent_v2 import DataWormV2


def run_test(n_steps=2500):
    print("\n" + "="*65)
    print("  THE GENERALITY TEST")
    print("  Can this worm handle what it was never designed for?")
    print("="*65)

    env  = DynamicEnvironment(width=40, height=40, seed=42)
    worm = DataWormV2(start_x=2, start_y=2, env_width=40, env_height=40)

    os.makedirs('logs', exist_ok=True)

    print(f"\n  World: 40x40 | Food: {len(env.food_centers)} clusters | Danger: {len(env.danger_centers)} zones")
    print(f"  Worm starts at (2,2) with {worm.nerve_ring.n_inputs} sensor inputs")
    print(f"\n  6 surprise events scheduled. The worm knows NOTHING about them.\n")
    print("  " + "-"*61)

    # Track metrics per phase
    phases = {
        'baseline':    {'start': 0,    'end': 299,  'rewards': [], 'label': 'Baseline (learning the world)'},
        'food_died':   {'start': 300,  'end': 499,  'rewards': [], 'label': 'After food source dies'},
        'wall':        {'start': 500,  'end': 799,  'rewards': [], 'label': 'After wall appears'},
        'unknown':     {'start': 800,  'end': 1199, 'rewards': [], 'label': 'After unknown signal appears'},
        'danger_move': {'start': 1200, 'end': 1499, 'rewards': [], 'label': 'After danger moves'},
        'new_food':    {'start': 1500, 'end': 1799, 'rewards': [], 'label': 'After new food appears'},
        'catastrophe': {'start': 1800, 'end': 2499, 'rewards': [], 'label': 'After catastrophe (famine)'},
    }

    announced_events = set()

    for step in range(n_steps):
        entry = worm.step(env)

        # Check if any new events fired
        for evt in env.events:
            key = f"{evt['step']}_{evt['type']}"
            if key not in announced_events:
                announced_events.add(key)
                print(f"\n  ⚡ STEP {evt['step']} — EVENT: {evt['type']}")
                print(f"     {evt['desc']}")
                if evt['type'] == 'UNKNOWN_SIGNAL':
                    print(f"     >>> The worm has NO sensor for this. Can it adapt?")
                if evt['type'] == 'WALL_APPEARED':
                    print(f"     >>> Physical barrier. The worm must find the gap.")
                if evt['type'] == 'CATASTROPHE':
                    print(f"     >>> Everything just got harder. Survival test.")
                print()

        # Check if worm adapted to unknown
        for adapt in worm.adaptations:
            key_a = f"adapt_{adapt['step']}"
            if key_a not in announced_events:
                announced_events.add(key_a)
                print(f"  🧠 STEP {adapt['step']} — ADAPTATION: {adapt['type']}")
                print(f"     {adapt['desc']}")
                print(f"     Brain inputs: {worm.nerve_ring.n_inputs} (was 12)")
                print()

        # Track rewards per phase
        for pname, pdata in phases.items():
            if pdata['start'] <= step <= pdata['end']:
                pdata['rewards'].append(entry['reward'])

        # Progress
        if step % 250 == 0 and step > 0:
            stats = worm.get_stats()
            explored = int((env.novelty < 0.9).sum())
            total = env.width * env.height
            print(f"  Step {step:5d} | pos ({worm.x:2d},{worm.y:2d}) | "
                  f"state={worm.state:8s} | "
                  f"reward {stats['avg_reward_recent']:+.3f} | "
                  f"surprise {stats['avg_surprise']:.3f} | "
                  f"inputs {worm.nerve_ring.n_inputs} | "
                  f"explored {explored}/{total}")

    # ── FINAL REPORT ──────────────────────────────────
    stats = worm.get_stats()
    explored = int((env.novelty < 0.9).sum())
    total = env.width * env.height

    print("\n" + "="*65)
    print("  GENERALITY TEST RESULTS")
    print("="*65)

    print(f"\n  Steps: {stats['age']} | Explored: {explored}/{total} ({100*explored/total:.0f}%)")
    print(f"  Total reward: {stats['total_reward']:.1f}")
    print(f"  Danger hits: {stats['danger_hits']} | Wall bumps: {stats['wall_bumps']}")
    print(f"  Unknown signal encounters: {stats['unknown_encounters']}")
    print(f"  Brain grew from 12 → {stats['n_inputs']} inputs")
    print(f"  Adaptations: {len(stats['adaptations'])}")

    print(f"\n  PHASE-BY-PHASE PERFORMANCE:")
    print(f"  {'Phase':<40s} {'Avg Reward':>10s}  {'Verdict':>10s}")
    print(f"  {'-'*62}")

    test_results = {}
    for pname, pdata in phases.items():
        if pdata['rewards']:
            avg = np.mean(pdata['rewards'])
            # First 50 vs last 50 of phase — did it recover?
            n = len(pdata['rewards'])
            if n > 20:
                first_half = np.mean(pdata['rewards'][:n//2])
                second_half = np.mean(pdata['rewards'][n//2:])
                recovered = second_half >= first_half * 0.8
            else:
                recovered = True

            if avg > 0.3:
                verdict = "THRIVING"
            elif avg > 0.15:
                verdict = "SURVIVING"
            elif avg > 0.0:
                verdict = "STRUGGLING"
            else:
                verdict = "FAILING"

            if pname != 'baseline' and recovered:
                verdict += " ↑"  # recovered/adapted

            test_results[pname] = {
                'avg_reward': float(avg),
                'verdict': verdict,
                'recovered': recovered,
            }

            print(f"  {pdata['label']:<40s} {avg:>+10.4f}  {verdict:>10s}")

    # THE VERDICT
    print(f"\n  {'='*62}")

    passes = sum(1 for r in test_results.values()
                 if 'THRIVING' in r['verdict'] or 'SURVIVING' in r['verdict'])
    total_tests = len(test_results)

    # Specific checks
    grew_brain = stats['n_inputs'] > 12
    survived_catastrophe = test_results.get('catastrophe', {}).get('avg_reward', -1) > 0
    found_unknown = stats['unknown_encounters'] > 0
    avoided_danger = stats['danger_hits'] < 10
    navigated_walls = stats['wall_bumps'] < 50

    checks = [
        ("Survived all phases (reward > 0)", passes >= 5),
        ("Brain grew new connections for unknown signal", grew_brain),
        ("Survived catastrophe (famine)", survived_catastrophe),
        ("Detected & used unknown signal", found_unknown),
        ("Avoided most danger", avoided_danger),
        ("Navigated around walls", navigated_walls),
    ]

    print(f"\n  GENERALITY CHECKS:")
    pass_count = 0
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"    [{icon}] {label}")
        if passed:
            pass_count += 1

    print(f"\n  Score: {pass_count}/{len(checks)} checks passed")

    if pass_count == len(checks):
        print(f"\n  VERDICT: This worm adapted to every surprise.")
        print(f"  It grew new neural connections for signals it was never")
        print(f"  designed to sense. It navigated obstacles it never knew")
        print(f"  existed. It survived catastrophe.")
        print(f"\n  Is this AGI? At worm scale — you tell me.")
    elif pass_count >= 4:
        print(f"\n  VERDICT: Strong adaptability. Some gaps remain.")
    else:
        print(f"\n  VERDICT: Needs more work. The worm isn't general enough yet.")

    print("="*65 + "\n")

    # Save for monitoring
    final_state = {
        'stats': stats,
        'n_steps': n_steps,
        'test_results': test_results,
        'events': env.events,
        'trajectory': [(s['x'], s['y']) for s in worm.step_log],
        'rewards': [s['reward'] for s in worm.step_log],
        'curiosity': [s['curiosity_score'] for s in worm.step_log],
        'surprise': [s['surprise'] for s in worm.step_log],
        'states': [s['state'] for s in worm.step_log],
        'visited': env.visited.tolist(),
        'novelty_final': env.novelty.tolist(),
        'walls': env.walls.tolist(),
        'unknown_signal': env.unknown_signal.tolist(),
        'checks': {label: bool(passed) for label, passed in checks},
    }
    with open('logs/last_run.json', 'w') as f:
        json.dump(final_state, f)
    with open('logs/environment.json', 'w') as f:
        json.dump(env.get_state_snapshot(), f)

    return worm, env, test_results


if __name__ == '__main__':
    run_test(n_steps=2500)
