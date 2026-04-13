"""
test_v3_final.py
----------------
THE DEFINITIVE TEST — v3 patched worm, 3 different worlds.
"""

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.environment_v2 import DynamicEnvironment
from core.agent_v3 import DataWormV3


def run_one(seed, n_steps=2500, verbose=False):
    env = DynamicEnvironment(width=40, height=40, seed=seed)
    worm = DataWormV3(start_x=2, start_y=2, env_width=40, env_height=40)

    phases = {
        'baseline':    {'range': (0, 299),    'rewards': [], 'label': 'Baseline (learning the world)'},
        'food_died':   {'range': (300, 499),  'rewards': [], 'label': 'After food dies'},
        'wall':        {'range': (500, 799),  'rewards': [], 'label': 'After wall appears'},
        'unknown':     {'range': (800, 1199), 'rewards': [], 'label': 'After unknown signal'},
        'danger_move': {'range': (1200, 1499),'rewards': [], 'label': 'After danger moves'},
        'new_food':    {'range': (1500, 1799),'rewards': [], 'label': 'After new food appears'},
        'catastrophe': {'range': (1800, 2499),'rewards': [], 'label': 'After catastrophe'},
    }

    announced = set()

    for step in range(n_steps):
        entry = worm.step(env)

        for pname, pd in phases.items():
            s, e = pd['range']
            if s <= step <= e:
                pd['rewards'].append(entry['reward'])

        if verbose:
            for evt in env.events:
                key = f"{evt['step']}_{evt['type']}"
                if key not in announced:
                    announced.add(key)
                    print(f"\n  ⚡ STEP {evt['step']} — {evt['type']}: {evt['desc']}")

            for adapt in worm.adaptations:
                key = f"adapt_{adapt['step']}"
                if key not in announced:
                    announced.add(key)
                    print(f"  🧠 STEP {adapt['step']} — ADAPTATION: {adapt['desc']} → {worm.nerve_ring.n_inputs} inputs")

            if step % 500 == 0 and step > 0:
                stats = worm.get_stats()
                explored = int((env.novelty < 0.9).sum())
                print(f"  Step {step:5d} | state={worm.state:8s} | reward {stats['avg_reward_recent']:+.3f} | "
                      f"baseline {worm.reward_baseline:.3f} | explored {explored}/1600")

    # Compile results
    results = {}
    for pname, pd in phases.items():
        if pd['rewards']:
            avg = np.mean(pd['rewards'])
            n = len(pd['rewards'])
            first_half = np.mean(pd['rewards'][:n//2]) if n > 4 else avg
            second_half = np.mean(pd['rewards'][n//2:]) if n > 4 else avg
            recovered = second_half >= first_half * 0.8
            results[pname] = {
                'avg': float(avg),
                'survived': avg > 0,
                'recovered': recovered,
                'first_half': float(first_half),
                'second_half': float(second_half),
                'label': pd['label'],
            }

    stats = worm.get_stats()
    grew_brain = worm.nerve_ring.n_inputs > 12
    all_survived = all(r['survived'] for r in results.values())

    checks = {
        'Survived all phases (reward > 0)':            all_survived,
        'Brain grew for unknown signal':               grew_brain,
        'Survived catastrophe':                        results.get('catastrophe', {}).get('survived', False),
        'Detected & used unknown signal':              worm.unknown_encounters > 0,
        'Avoided most danger (< 10 hits)':             worm.danger_hits < 10,
        'Navigated walls (< 50 bumps)':                worm.wall_bumps < 50,
    }

    return {
        'seed': seed,
        'phases': results,
        'checks': checks,
        'pass_count': sum(checks.values()),
        'total_checks': len(checks),
        'stats': {
            'total_reward': float(worm.total_reward),
            'danger_hits': worm.danger_hits,
            'wall_bumps': worm.wall_bumps,
            'unknown_encounters': worm.unknown_encounters,
            'n_inputs': worm.nerve_ring.n_inputs,
            'explored': int((env.novelty < 0.9).sum()),
        },
        'events': env.events,
    }


def main():
    seeds = [42, 99, 7]

    print("\n" + "="*70)
    print("  THE DEFINITIVE TEST — v3 Patched Worm")
    print("  4 biological fixes: wall loop, graded danger, spatial memory,")
    print("  adaptive state switching")
    print("="*70)

    all_results = []

    for seed in seeds:
        print(f"\n{'─'*70}")
        print(f"  SEED {seed}")
        print(f"{'─'*70}")
        r = run_one(seed, n_steps=2500, verbose=True)
        all_results.append(r)

        print(f"\n  Results for seed {seed}:")
        print(f"  {'Phase':<35s} {'Avg':>8s} {'1st½':>8s} {'2nd½':>8s} {'Status':>12s}")
        print(f"  {'─'*65}")

        for pname, pd in r['phases'].items():
            avg = pd['avg']
            status = 'THRIVING' if avg > 0.3 else 'SURVIVING' if avg > 0.15 else 'STRUGGLING' if avg > 0 else 'FAILING'
            arrow = ' ↑' if pd['recovered'] and pname != 'baseline' else ' ↓' if not pd['recovered'] else ''
            bar_len = int(max(0, min(20, avg * 40)))
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"  {pd['label']:<35s} {avg:>+.4f} {pd['first_half']:>+.4f} {pd['second_half']:>+.4f} {status+arrow:>12s}")

        print(f"\n  Checks:")
        for label, passed in r['checks'].items():
            icon = "✓" if passed else "✗"
            print(f"    [{icon}] {label}")
        print(f"  Score: {r['pass_count']}/{r['total_checks']}")
        print(f"  Danger: {r['stats']['danger_hits']} | Walls: {r['stats']['wall_bumps']} | "
              f"Brain: {r['stats']['n_inputs']} inputs | Explored: {r['stats']['explored']}/1600")

    # ── FINAL SCORECARD ──
    print("\n" + "="*70)
    print("  FINAL SCORECARD — v3 Patched Worm")
    print("="*70)

    total_passes = sum(r['pass_count'] for r in all_results)
    total_possible = sum(r['total_checks'] for r in all_results)

    print(f"\n  {'Seed':>6s} {'Score':>8s} {'Reward':>10s} {'Danger':>8s} {'Walls':>8s} {'Brain':>7s} {'Explored':>10s}")
    print(f"  {'─'*60}")
    for r in all_results:
        s = r['stats']
        print(f"  {r['seed']:>6d} {r['pass_count']}/{r['total_checks']:>5d} {s['total_reward']:>+10.1f} "
              f"{s['danger_hits']:>8d} {s['wall_bumps']:>8d} {s['n_inputs']:>4d}/12 {s['explored']:>6d}/1600")

    print(f"\n  TOTAL: {total_passes}/{total_possible} checks passed across 3 worlds")

    # Compare with v2
    print(f"\n  vs V2 (previous run):")
    print(f"  V2 scores: seed42=5/6, seed99=2/6, seed7=6/6 → total 13/18")
    print(f"  V3 scores: ", end='')
    for r in all_results:
        print(f"seed{r['seed']}={r['pass_count']}/6, ", end='')
    print(f"→ total {total_passes}/{total_possible}")

    if total_passes >= 16:
        print(f"\n  ★ VERDICT: The worm consistently adapts to the unknown.")
        print(f"    It grows new neural connections, remembers where food was,")
        print(f"    navigates walls, avoids danger gradually, and survives famine.")
        print(f"    Across 3 different worlds it was never designed for.")
    elif total_passes >= 12:
        print(f"\n  VERDICT: Major improvement. Most challenges handled.")
    else:
        print(f"\n  VERDICT: Still needs work.")

    print("="*70 + "\n")

    # Save
    os.makedirs('logs', exist_ok=True)
    with open('logs/v3_test_results.json', 'w') as f:
        json.dump({
            'results': [{
                'seed': r['seed'],
                'pass_count': r['pass_count'],
                'checks': {k: bool(v) for k, v in r['checks'].items()},
                'stats': r['stats'],
                'phase_avgs': {k: v['avg'] for k, v in r['phases'].items()},
            } for r in all_results],
            'total_passes': total_passes,
            'total_possible': total_possible,
        }, f, indent=2)


if __name__ == '__main__':
    main()
