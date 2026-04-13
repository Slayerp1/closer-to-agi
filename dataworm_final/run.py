"""
run.py
------
Run the DataWorm.

Usage:
  python run.py                  # run 1000 steps, print live
  python run.py --steps 5000     # run more steps
  python run.py --steps 1 --verbose   # single step, full detail (the 1mm test)

This logs everything to logs/ so the monitor can read it.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from core.environment import DataEnvironment
from core.agent import DataWorm


def run(n_steps=1000, verbose=False, log_interval=10):
    print("\n" + "="*60)
    print("  DATAWORM — curiosity-driven data explorer")
    print("  A mind grown from a single drive: move toward the unknown")
    print("="*60 + "\n")

    # ── Initialize ──────────────────────────────────────
    env   = DataEnvironment(width=40, height=40, seed=42)
    worm  = DataWorm(start_x=2, start_y=2, env_width=40, env_height=40)

    os.makedirs('logs', exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/run_{run_id}.jsonl'

    print(f"Environment : 40×40 data space")
    print(f"Food sources: {len(env.food_centers)} rich data clusters")
    print(f"Danger zones: {len(env.danger_centers)} corrupted regions")
    print(f"Agent start : ({worm.x}, {worm.y})")
    print(f"Log file    : {log_path}")
    print(f"\nRunning {n_steps} steps...\n")

    # Save environment state for monitor
    with open('logs/environment.json', 'w') as f:
        json.dump(env.get_state_snapshot(), f)

    # ── THE LOOP ────────────────────────────────────────
    with open(log_path, 'w') as log_file:
        for step in range(n_steps):

            # ONE STEP — this is everything
            entry = worm.step(env)

            # Write to log
            log_file.write(json.dumps(entry) + '\n')
            log_file.flush()

            # Print progress
            if verbose or step == 0:
                print(f"\n── Step {entry['step']} ──────────────────────")
                print(f"  Position  : ({entry['x']}, {entry['y']})")
                print(f"  Action    : {entry['action']} facing {entry['direction']}")
                print(f"  Curiosity : {entry['curiosity_score']}")
                print(f"  Reward    : {entry['reward']}")
                print(f"  Reflex    : {'FIRED — DANGER DETECTED' if entry['reflex_fired'] else 'no'}")
                print(f"  Richness  : {entry['sensors']['richness_here']}")
                print(f"  Danger    : {entry['sensors']['danger_here']}")
                print(f"  Novelty   : {entry['sensors']['novelty_here']}")
                print(f"  W.change  : {entry['weight_change']}")

                if step == 0:
                    print("\n  >>> FIRST STEP TAKEN. THE WORM IS ALIVE. <<<")
                    if entry['x'] != 2 or entry['y'] != 2:
                        print("  >>> IT MOVED ON ITS OWN. NO PROMPT. NO INSTRUCTION. <<<")

            elif step % log_interval == 0:
                stats = worm.get_stats()
                bar_len = 30
                explored = int(bar_len * (env.width * env.height - (env.novelty > 0.8).sum()) / (env.width * env.height))
                bar = '█' * explored + '░' * (bar_len - explored)
                print(f"  Step {step:5d} | pos ({worm.x:2d},{worm.y:2d}) | "
                      f"reward {stats['avg_reward_recent']:+.3f} | "
                      f"curiosity {stats['avg_curiosity']:.3f} | "
                      f"explored [{bar}]")

    # ── Final summary ────────────────────────────────────
    stats = worm.get_stats()
    cells_explored = int((env.novelty < 0.9).sum())
    total_cells    = env.width * env.height

    print("\n" + "="*60)
    print("  FINAL REPORT")
    print("="*60)
    print(f"  Steps taken   : {stats['age']}")
    print(f"  Total reward  : {stats['total_reward']}")
    print(f"  Data found    : {stats['data_found']}")
    print(f"  Novelty seen  : {stats['novelty_seen']}")
    print(f"  Danger hits   : {stats['danger_hits']}")
    print(f"  Reflex rate   : {stats['reflex_rate']*100:.1f}% of steps")
    print(f"  Cells explored: {cells_explored}/{total_cells} ({100*cells_explored/total_cells:.1f}%)")
    print(f"  Weight growth : {stats['weight_stats']['W1_mean']:.5f} → brain is learning")
    print(f"\n  Log saved: {log_path}")
    print("="*60 + "\n")

    # Save final agent state for monitor
    final_state = {
        'stats': stats,
        'run_id': run_id,
        'n_steps': n_steps,
        'trajectory': [(s['x'], s['y']) for s in worm.step_log],
        'rewards': [s['reward'] for s in worm.step_log],
        'curiosity': [s['curiosity_score'] for s in worm.step_log],
        'visited': env.visited.tolist(),
        'novelty_final': env.novelty.tolist(),
    }
    with open('logs/last_run.json', 'w') as f:
        json.dump(final_state, f)

    return worm, env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',   type=int,  default=1000)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    run(n_steps=args.steps, verbose=args.verbose)
