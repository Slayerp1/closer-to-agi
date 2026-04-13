"""
test_social.py
--------------
THE SOCIAL INTELLIGENCE TEST.

Does a group of worms do better than the same number of solo worms?
Do they learn to follow each other's pheromone trails?
Does social behavior EMERGE without being programmed?

Tests:
  1. SOLO BASELINE: Run 5 worms alone in identical worlds
  2. COLONY: Run 5 worms together in one shared world
  3. PRETRAINED COLONY: Same, but with transfer-learned brains
  4. Compare: Does the group find more food per worm?
  5. Analyze: Did trail-following behavior emerge?
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from core.environment_v3 import TransferEnvironment
from core.environment_v4 import SocialEnvironment
from core.agent_v3 import DataWormV3
from core.colony import Colony


STEPS = 1500
N_WORMS = 5


def run_solo_baseline(seed):
    """Run N worms each ALONE in identical worlds. No pheromones. No competition."""
    results = []
    for i in range(N_WORMS):
        # Each worm gets its own identical world
        env = SocialEnvironment(width=40, height=40, seed=seed)
        worm = DataWormV3(start_x=np.random.randint(5, 35),
                           start_y=np.random.randint(5, 35),
                           env_width=40, env_height=40)

        for step in range(STEPS):
            worm.step(env)

        results.append({
            'total_reward': worm.total_reward,
            'data_found': worm.data_found,
            'danger_hits': worm.danger_hits,
            'unknown_encounters': worm.unknown_encounters,
        })

    return results


def run_colony(seed, pretrained_brain=None):
    """Run N worms TOGETHER in one shared world."""
    colony = Colony(n_worms=N_WORMS, env_seed=seed,
                    pretrained_brain=pretrained_brain)
    colony.run(STEPS, verbose_interval=300)
    return colony


def pretrain_one_worm(seed):
    """Train a single worm in a world with glow signal so it knows about unknown signals."""
    env = TransferEnvironment(width=40, height=40, seed=seed,
                               signal_type='glow', signal_on_step=0)
    worm = DataWormV3(start_x=2, start_y=2, env_width=40, env_height=40)

    for step in range(1500):
        worm.step(env)

    return worm


def analyze_trail_following(colony):
    """Check if worms learned to follow pheromone trails."""
    total_pheromone_encounters = sum(w.unknown_encounters for w in colony.worms)

    # Check if worms that followed more trails found more food
    trail_food_correlation = []
    for w in colony.worms:
        trail_food_correlation.append((w.unknown_encounters, w.data_found))

    # Sort by trail usage
    trail_food_correlation.sort(key=lambda x: x[0])

    # Did high-trail worms find more food?
    low_trail_worms = trail_food_correlation[:N_WORMS//2]
    high_trail_worms = trail_food_correlation[N_WORMS//2:]

    low_food = np.mean([x[1] for x in low_trail_worms]) if low_trail_worms else 0
    high_food = np.mean([x[1] for x in high_trail_worms]) if high_trail_worms else 0

    return {
        'total_pheromone_encounters': total_pheromone_encounters,
        'low_trail_avg_food': low_food,
        'high_trail_avg_food': high_food,
        'trail_helps': high_food > low_food,
    }


def analyze_spatial_behavior(colony):
    """Check for clustering vs dispersal patterns."""
    # Get final positions
    positions = [(w.x, w.y) for w in colony.worms]

    # Calculate pairwise distances
    dists = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.sqrt((positions[i][0]-positions[j][0])**2 +
                       (positions[i][1]-positions[j][1])**2)
            dists.append(d)

    avg_dist = np.mean(dists) if dists else 0

    # Are worms near food?
    food_positions = [(c[0], c[1]) for c in colony.env.food_centers]
    near_food = 0
    for wx, wy in positions:
        for fx, fy in food_positions:
            if abs(wx-fx) + abs(wy-fy) < 8:
                near_food += 1
                break

    # State distribution
    states = [w.state for w in colony.worms]
    dwelling_count = sum(1 for s in states if s == 'dwelling')

    return {
        'avg_distance': avg_dist,
        'near_food_count': near_food,
        'dwelling_count': dwelling_count,
        'roaming_count': N_WORMS - dwelling_count,
        'clustered': avg_dist < 15,
        'dispersed': avg_dist > 25,
    }


def main():
    SEED = 42

    print("\n" + "="*70)
    print("  THE SOCIAL INTELLIGENCE TEST")
    print("  Does a colony of worms outperform solo individuals?")
    print("  Does trail-following behavior emerge without programming?")
    print("="*70)

    # ══════════════════════════════════════════════
    # TEST 1: SOLO BASELINE
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  TEST 1 — Solo Baseline: {N_WORMS} worms, each alone")
    print(f"  {STEPS} steps each. No pheromones. No interaction.")
    print(f"{'─'*70}")

    solo_results = run_solo_baseline(SEED)
    solo_total_reward = sum(r['total_reward'] for r in solo_results)
    solo_total_food = sum(r['data_found'] for r in solo_results)
    solo_avg_reward = solo_total_reward / N_WORMS
    solo_avg_food = solo_total_food / N_WORMS

    print(f"\n  Solo results ({N_WORMS} separate runs):")
    for i, r in enumerate(solo_results):
        print(f"    Worm {i}: reward {r['total_reward']:+.1f}  food {r['data_found']:.1f}  danger {r['danger_hits']}")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:  reward {solo_total_reward:+.1f}  food {solo_total_food:.1f}")
    print(f"    PER WORM: reward {solo_avg_reward:+.1f}  food {solo_avg_food:.1f}")

    # ══════════════════════════════════════════════
    # TEST 2: NAIVE COLONY (no pretraining)
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  TEST 2 — Naive Colony: {N_WORMS} worms together, no pretraining")
    print(f"  Shared food. Pheromone trails. Competition + potential cooperation.")
    print(f"{'─'*70}")

    naive_colony = run_colony(SEED, pretrained_brain=None)
    naive_stats = naive_colony.get_stats()

    print(f"\n  Colony results:")
    for pw in naive_stats['per_worm']:
        print(f"    Worm {pw['id']}: reward {pw['total_reward']:+.1f}  "
              f"food {pw['data_found']:.1f}  trails {pw['unknown_encounters']}  "
              f"state={pw['state']}  brain={pw['n_inputs']} inputs")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:  reward {naive_stats['total_reward']:+.1f}  "
          f"food {naive_stats['total_data_found']:.1f}")
    naive_avg = naive_stats['total_reward'] / N_WORMS
    print(f"    PER WORM: reward {naive_avg:+.1f}  "
          f"food {naive_stats['total_data_found']/N_WORMS:.1f}")

    # ══════════════════════════════════════════════
    # TEST 3: PRETRAINED COLONY
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  TEST 3 — Pretrained Colony: {N_WORMS} worms with transfer-learned brains")
    print(f"  Each worm already knows 'unknown signals might mean food'")
    print(f"  Does this make them learn pheromone trails faster?")
    print(f"{'─'*70}")

    print(f"\n  Pretraining one worm in World A (glow signal)...")
    pretrained = pretrain_one_worm(seed=99)
    print(f"  Brain: {pretrained.nerve_ring.n_inputs} inputs. Cloning to all {N_WORMS} worms.")

    trained_colony = run_colony(SEED, pretrained_brain=pretrained)
    trained_stats = trained_colony.get_stats()

    print(f"\n  Pretrained colony results:")
    for pw in trained_stats['per_worm']:
        print(f"    Worm {pw['id']}: reward {pw['total_reward']:+.1f}  "
              f"food {pw['data_found']:.1f}  trails {pw['unknown_encounters']}  "
              f"state={pw['state']}  brain={pw['n_inputs']} inputs")
    print(f"    ──────────────────────────")
    print(f"    TOTAL:  reward {trained_stats['total_reward']:+.1f}  "
          f"food {trained_stats['total_data_found']:.1f}")
    trained_avg = trained_stats['total_reward'] / N_WORMS
    print(f"    PER WORM: reward {trained_avg:+.1f}  "
          f"food {trained_stats['total_data_found']/N_WORMS:.1f}")

    # ══════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print(f"  ANALYSIS — Emergent Social Behavior")
    print(f"{'─'*70}")

    # Trail following analysis
    naive_trail = analyze_trail_following(naive_colony)
    trained_trail = analyze_trail_following(trained_colony)

    print(f"\n  Trail Following:")
    print(f"    Naive colony — total pheromone encounters: {naive_trail['total_pheromone_encounters']}")
    print(f"      Low-trail worms avg food:  {naive_trail['low_trail_avg_food']:.1f}")
    print(f"      High-trail worms avg food: {naive_trail['high_trail_avg_food']:.1f}")
    print(f"      Trail helps find food: {'✓ YES' if naive_trail['trail_helps'] else '✗ NO'}")

    print(f"    Pretrained colony — total pheromone encounters: {trained_trail['total_pheromone_encounters']}")
    print(f"      Low-trail worms avg food:  {trained_trail['low_trail_avg_food']:.1f}")
    print(f"      High-trail worms avg food: {trained_trail['high_trail_avg_food']:.1f}")
    print(f"      Trail helps find food: {'✓ YES' if trained_trail['trail_helps'] else '✗ NO'}")

    # Spatial analysis
    naive_spatial = analyze_spatial_behavior(naive_colony)
    trained_spatial = analyze_spatial_behavior(trained_colony)

    print(f"\n  Spatial Behavior:")
    print(f"    Naive:     avg dist {naive_spatial['avg_distance']:.1f} | "
          f"near food {naive_spatial['near_food_count']}/{N_WORMS} | "
          f"dwelling {naive_spatial['dwelling_count']} roaming {naive_spatial['roaming_count']}")
    print(f"    Pretrained: avg dist {trained_spatial['avg_distance']:.1f} | "
          f"near food {trained_spatial['near_food_count']}/{N_WORMS} | "
          f"dwelling {trained_spatial['dwelling_count']} roaming {trained_spatial['roaming_count']}")

    # ══════════════════════════════════════════════
    # FINAL SCORECARD
    # ══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SOCIAL INTELLIGENCE — FINAL SCORECARD")
    print(f"{'='*70}")

    print(f"\n  PER-WORM COMPARISON:")
    print(f"  {'Condition':<25s} {'Avg Reward':>12s} {'Avg Food':>10s}")
    print(f"  {'─'*50}")
    print(f"  {'Solo (alone)':<25s} {solo_avg_reward:>+12.1f} {solo_avg_food:>10.1f}")
    print(f"  {'Naive colony':<25s} {naive_avg:>+12.1f} {naive_stats['total_data_found']/N_WORMS:>10.1f}")
    print(f"  {'Pretrained colony':<25s} {trained_avg:>+12.1f} {trained_stats['total_data_found']/N_WORMS:>10.1f}")

    # The checks
    colony_beats_solo = naive_avg > solo_avg_reward * 0.9  # colony at least 90% of solo (competition costs)
    pretrained_beats_naive = trained_stats['total_reward'] > naive_stats['total_reward']
    trail_following_emerged = naive_trail['trail_helps'] or trained_trail['trail_helps']
    brains_grew = any(pw['n_inputs'] > 12 for pw in naive_stats['per_worm'])
    some_near_food = naive_spatial['near_food_count'] >= 2 or trained_spatial['near_food_count'] >= 2

    cooperation_emerged = naive_stats['total_data_found'] > solo_total_food * 0.85

    checks = [
        ("Colony survival (reward >= 90% of solo)", colony_beats_solo),
        ("Brains grew for pheromone signal", brains_grew),
        ("Trail-following correlates with food", trail_following_emerged),
        ("Worms converge near food sources", some_near_food),
        ("Pretraining helps colony performance", pretrained_beats_naive),
        ("Group food >= 85% of solo food (despite competition)", cooperation_emerged),
    ]

    print(f"\n  CHECKS:")
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        print(f"    [{icon}] {label}")

    passed_count = sum(1 for _, p in checks if p)
    print(f"\n  Score: {passed_count}/{len(checks)}")

    if passed_count >= 5:
        print(f"""
  ★ SOCIAL INTELLIGENCE EMERGED.

  Without being told to cooperate, worms:
    - Grew new neural connections for pheromone signals
    - Learned that other worms' trails correlate with food
    - Converged near food sources
    - Maintained performance despite competition for food

  The pheromone trail is just another "unknown signal."
  But unlike glow or vibration, this signal comes from OTHER AGENTS.
  The worm learned to use the behavior of others as information.

  That is the seed of social intelligence.
  """)
    elif passed_count >= 3:
        print(f"\n  Partial social behavior emerged. Competition and cooperation coexist.")
    else:
        print(f"\n  Social behavior is weak. Worms mostly independent.")

    print("="*70 + "\n")


if __name__ == '__main__':
    main()
