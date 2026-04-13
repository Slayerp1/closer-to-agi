"""
test_deliberation.py
--------------------
Does thinking before acting help?

Compare:
  - v4 worm (reactive only — no prediction)
  - v5 worm (deliberative — predicts outcomes before choosing)

In the same challenging world with dynamic events.
If v5 beats v4, the worm benefits from imagining the future.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.environment_v2 import DynamicEnvironment
from core.agent_v4 import DataWormV4
from core.agent_v5 import DataWormV5


STEPS = 2500


def run_worm(worm_class, seed, label):
    env = DynamicEnvironment(width=40, height=40, seed=seed)
    worm = worm_class(start_x=2, start_y=2, env_width=40, env_height=40)

    rewards = []
    cumulative = 0
    checkpoints = {}

    for step in range(STEPS):
        entry = worm.step(env)
        rewards.append(entry['reward'])
        cumulative += entry['reward']

        if step + 1 in [500, 1000, 1500, 2000, 2500]:
            checkpoints[step + 1] = {
                'cumulative': cumulative,
                'avg_recent': np.mean(rewards[-100:]),
            }

    stats = worm.get_stats()
    explored = int((env.novelty < 0.9).sum())

    return {
        'label': label,
        'total_reward': cumulative,
        'avg_reward': np.mean(rewards),
        'checkpoints': checkpoints,
        'explored': explored,
        'danger_hits': worm.danger_hits,
        'wall_bumps': worm.wall_bumps,
        'stats': stats,
        'rewards': rewards,
    }


def main():
    print("\n" + "="*70)
    print("  THE DELIBERATION TEST")
    print("  Does imagining the future before acting help?")
    print("="*70)

    results = {}

    for seed in [42, 99, 7]:
        print(f"\n{'─'*70}")
        print(f"  WORLD SEED {seed}")
        print(f"{'─'*70}")

        # Reactive worm (v4 — plastic drives, no prediction)
        r_reactive = run_worm(DataWormV4, seed, f"Reactive (v4) seed={seed}")

        # Deliberative worm (v5 — plastic drives + prediction)
        r_delib = run_worm(DataWormV5, seed, f"Deliberative (v5) seed={seed}")

        results[seed] = {'reactive': r_reactive, 'deliberative': r_delib}

        print(f"\n  {'Metric':<25s} {'Reactive (v4)':>14s} {'Deliberative (v5)':>18s} {'Δ':>10s}")
        print(f"  {'─'*68}")

        for metric in ['total_reward', 'avg_reward', 'explored', 'danger_hits', 'wall_bumps']:
            rv = r_reactive[metric]
            dv = r_delib[metric]
            delta = dv - rv if isinstance(dv, (int, float)) else 0
            rv_s = f"{rv:+.1f}" if isinstance(rv, float) else str(rv)
            dv_s = f"{dv:+.1f}" if isinstance(dv, float) else str(dv)
            better = '✓' if (delta > 0 and metric in ['total_reward', 'avg_reward', 'explored']) or \
                            (delta < 0 and metric in ['danger_hits', 'wall_bumps']) else ''
            print(f"  {metric:<25s} {rv_s:>14s} {dv_s:>18s} {delta:>+10.1f} {better}")

        # Prediction stats
        ds = r_delib['stats']
        if 'prediction' in ds:
            ps = ds['prediction']
            print(f"\n  Prediction stats:")
            print(f"    Avg prediction error:  {ps['avg_prediction_error']:.4f}")
            print(f"    Predictions improving: {ps['prediction_improving']}")
            print(f"    Prediction weight:     {ds.get('prediction_weight', 0):.4f}")
            print(f"    Prediction overrides:  {ds.get('prediction_overrides', 0)}")

        # Performance over time
        print(f"\n  Cumulative reward over time:")
        for step in [500, 1000, 1500, 2000, 2500]:
            rc = r_reactive['checkpoints'].get(step, {}).get('cumulative', 0)
            dc = r_delib['checkpoints'].get(step, {}).get('cumulative', 0)
            lead = "v5 leads" if dc > rc else "v4 leads"
            print(f"    Step {step:5d}: v4={rc:+8.1f}  v5={dc:+8.1f}  ({lead} by {abs(dc-rc):.1f})")

    # ══════════════════════════════════════════════
    # FINAL SCORECARD
    # ══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  DELIBERATION TEST — FINAL SCORECARD")
    print(f"{'='*70}")

    v5_wins_reward = 0
    v5_wins_safety = 0
    v5_wins_exploration = 0

    for seed, r in results.items():
        rr = r['reactive']
        rd = r['deliberative']

        if rd['total_reward'] > rr['total_reward']:
            v5_wins_reward += 1
        if rd['danger_hits'] <= rr['danger_hits']:
            v5_wins_safety += 1
        if rd['explored'] >= rr['explored']:
            v5_wins_exploration += 1

    checks = [
        ("Deliberative beats reactive in total reward (2+ of 3 worlds)", v5_wins_reward >= 2),
        ("Deliberative is safer (fewer danger hits)", v5_wins_safety >= 2),
        ("Deliberative explores at least as much", v5_wins_exploration >= 2),
        ("Prediction error decreases over time", any(
            r['deliberative']['stats'].get('prediction', {}).get('prediction_improving', False)
            for r in results.values()
        )),
        ("Prediction overrides reactive decisions", any(
            r['deliberative']['stats'].get('prediction_overrides', 0) > 10
            for r in results.values()
        )),
    ]

    print(f"\n  {'Metric':<20s}  {'v5 wins':>8s}/3 worlds")
    print(f"  {'─'*35}")
    print(f"  {'Total reward':<20s}  {v5_wins_reward:>8d}")
    print(f"  {'Safety':<20s}  {v5_wins_safety:>8d}")
    print(f"  {'Exploration':<20s}  {v5_wins_exploration:>8d}")

    print(f"\n  CHECKS:")
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"    [{icon}] {label}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/{len(checks)}")

    if passed_count >= 4:
        print(f"""
  ★ DELIBERATION HELPS.

  The worm that imagines outcomes before acting outperforms
  the worm that only reacts. Its predictions improve over time.
  It starts reactive (like a newborn) and becomes deliberative
  (like an experienced animal).

  This is the transition from stimulus-response to anticipation.
  The seed of planning.
  """)
    elif passed_count >= 2:
        print(f"\n  Partial benefit. Deliberation helps in some conditions.")
    else:
        print(f"\n  Deliberation not yet beneficial. Predictions too inaccurate.")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
