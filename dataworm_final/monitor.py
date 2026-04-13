"""
monitor.py
----------
Live monitoring dashboard for the DataWorm.

Run AFTER or ALONGSIDE run.py:
  python monitor.py

Shows:
  - World map (food, danger, where worm has been)
  - Curiosity signal over time
  - Reward signal over time
  - Weight learning curve
  - Live sensor readings
  - Step-by-step log tail

Requires: pip install matplotlib numpy
"""

import json
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime


def load_env():
    try:
        with open('logs/environment.json') as f:
            return json.load(f)
    except:
        return None


def load_last_run():
    try:
        with open('logs/last_run.json') as f:
            return json.load(f)
    except:
        return None


def load_latest_log():
    """Load most recent .jsonl run log."""
    logs = [f for f in os.listdir('logs') if f.startswith('run_') and f.endswith('.jsonl')]
    if not logs:
        return []
    latest = sorted(logs)[-1]
    entries = []
    with open(f'logs/{latest}') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    return entries


def generate_report(output_path='logs/monitor_report.png'):
    env_data = load_env()
    run_data = load_last_run()
    log      = load_latest_log()

    if not env_data or not log:
        print("No run data found. Run `python run.py` first.")
        return

    fig = plt.figure(figsize=(18, 12), facecolor='#0d0d0d')
    fig.suptitle('DataWorm — Curiosity-Driven Data Explorer', 
                 color='#e0e0e0', fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35,
                          left=0.05, right=0.97, top=0.93, bottom=0.05)

    dark_ax_props = dict(facecolor='#1a1a1a', labelcolor='#aaa', 
                         titlecolor='#e0e0e0')

    # ── 1. WORLD MAP ──────────────────────────────────────────
    ax_world = fig.add_subplot(gs[0:2, 0:2])
    ax_world.set_facecolor('#1a1a1a')
    ax_world.set_title('Data space — worm trajectory', color='#e0e0e0', fontsize=11)

    W = env_data['width']
    H = env_data['height']

    richness = np.array(env_data['richness'])
    danger   = np.array(env_data['danger'])
    visited  = np.array(run_data['visited']) if run_data else np.zeros((H, W))

    # Composite world view: richness (green) overlaid with danger (red)
    world_rgb = np.zeros((H, W, 3))
    world_rgb[:,:,0] = danger   * 0.8    # Red channel = danger
    world_rgb[:,:,1] = richness * 0.7    # Green channel = richness
    world_rgb[:,:,2] = 0.1               # slight blue tint

    # Darken unexplored areas
    explore_mask = (visited == 0).astype(float)
    for c in range(3):
        world_rgb[:,:,c] *= (1 - explore_mask * 0.7)

    ax_world.imshow(world_rgb, origin='upper', aspect='equal')

    # Draw trajectory
    if log:
        xs = [e['x'] for e in log]
        ys = [e['y'] for e in log]
        # Color trajectory by step (early=blue, late=yellow)
        n = len(xs)
        for i in range(1, min(n, 500)):  # draw up to 500 steps
            alpha = min(1.0, i / 100)
            frac  = i / n
            color = (frac, frac * 0.5, 1 - frac)
            ax_world.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], 
                         color=color, linewidth=0.8, alpha=alpha)

        # Mark start and end
        ax_world.plot(xs[0],  ys[0],  'o', color='#00ff88', ms=8, label='start', zorder=5)
        ax_world.plot(xs[-1], ys[-1], '*', color='#ffff00', ms=10, label='end',  zorder=5)

    # Mark food centers
    for cx, cy in env_data['food_centers']:
        ax_world.plot(cx, cy, 'P', color='#88ff88', ms=12, markeredgecolor='white', 
                     markeredgewidth=0.5, zorder=6)

    # Mark danger centers
    for cx, cy in env_data['danger_centers']:
        ax_world.plot(cx, cy, 'X', color='#ff4444', ms=10, markeredgecolor='white',
                     markeredgewidth=0.5, zorder=6)

    legend_elements = [
        mpatches.Patch(color='#00ff88', label='start'),
        mpatches.Patch(color='#ffff00', label='end'),
        mpatches.Patch(color='#88ff88', label='data clusters'),
        mpatches.Patch(color='#ff4444', label='danger zones'),
    ]
    ax_world.legend(handles=legend_elements, loc='lower right', 
                   facecolor='#2a2a2a', labelcolor='#ccc', fontsize=8)
    ax_world.tick_params(colors='#666')

    # ── 2. NOVELTY MAP ────────────────────────────────────────
    ax_novel = fig.add_subplot(gs[0:2, 2])
    ax_novel.set_facecolor('#1a1a1a')
    ax_novel.set_title('Novelty remaining\n(dark=explored)', color='#e0e0e0', fontsize=10)
    novelty_final = np.array(run_data['novelty_final']) if run_data else np.array(env_data['richness'])
    im = ax_novel.imshow(novelty_final, cmap='plasma', origin='upper', 
                         vmin=0, vmax=1, aspect='equal')
    plt.colorbar(im, ax=ax_novel, fraction=0.046, pad=0.04).ax.tick_params(colors='#aaa')
    ax_novel.tick_params(colors='#666')

    # ── 3. STATS PANEL ────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0:2, 3])
    ax_stats.set_facecolor('#1a1a1a')
    ax_stats.axis('off')
    ax_stats.set_title('Run statistics', color='#e0e0e0', fontsize=10)

    if run_data:
        s = run_data['stats']
        total_cells = W * H
        explored_cells = int((np.array(run_data['novelty_final']) < 0.9).sum())

        lines = [
            ('Steps',          str(s['age'])),
            ('Final pos',      str(s['position'])),
            ('Total reward',   f"{s['total_reward']:.2f}"),
            ('Avg reward',     f"{s['avg_reward_recent']:.4f}"),
            ('Data found',     f"{s['data_found']:.2f}"),
            ('Novelty seen',   f"{s['novelty_seen']:.2f}"),
            ('Danger hits',    str(s['danger_hits'])),
            ('Reflex rate',    f"{s['reflex_rate']*100:.1f}%"),
            ('Explored',       f"{explored_cells}/{total_cells} ({100*explored_cells/total_cells:.0f}%)"),
            ('W1 mean',        f"{s['weight_stats']['W1_mean']:.5f}"),
            ('W2 mean',        f"{s['weight_stats']['W2_mean']:.5f}"),
            ('Learning Δ',     f"{s['weight_stats']['recent_change']:.6f}"),
        ]
        for i, (label, val) in enumerate(lines):
            y_pos = 0.95 - i * 0.075
            ax_stats.text(0.05, y_pos, label + ':', transform=ax_stats.transAxes,
                         color='#888', fontsize=9, va='top')
            ax_stats.text(0.55, y_pos, val, transform=ax_stats.transAxes,
                         color='#e0e0e0', fontsize=9, va='top', fontweight='bold')

    # ── 4. REWARD OVER TIME ───────────────────────────────────
    ax_reward = fig.add_subplot(gs[2, 0])
    ax_reward.set_facecolor('#1a1a1a')
    ax_reward.set_title('Reward over time', color='#e0e0e0', fontsize=10)
    if run_data:
        rewards = run_data['rewards']
        n = len(rewards)
        # Rolling average
        window = max(1, n // 50)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax_reward.plot(rewards, color='#334', linewidth=0.5, alpha=0.4)
        ax_reward.plot(smoothed, color='#4fc3f7', linewidth=1.5)
        ax_reward.axhline(0, color='#666', linewidth=0.5, linestyle='--')
        ax_reward.fill_between(range(len(smoothed)), 0, smoothed,
                               where=[s > 0 for s in smoothed],
                               alpha=0.2, color='#4fc3f7')
        ax_reward.fill_between(range(len(smoothed)), 0, smoothed,
                               where=[s < 0 for s in smoothed],
                               alpha=0.2, color='#ff6b6b')
    ax_reward.tick_params(colors='#666')
    ax_reward.spines['bottom'].set_color('#333')
    ax_reward.spines['left'].set_color('#333')
    ax_reward.spines['top'].set_visible(False)
    ax_reward.spines['right'].set_visible(False)

    # ── 5. CURIOSITY OVER TIME ────────────────────────────────
    ax_curv = fig.add_subplot(gs[2, 1])
    ax_curv.set_facecolor('#1a1a1a')
    ax_curv.set_title('Curiosity signal', color='#e0e0e0', fontsize=10)
    if run_data:
        curiosity = run_data['curiosity']
        n = len(curiosity)
        window = max(1, n // 50)
        smoothed_c = np.convolve(curiosity, np.ones(window)/window, mode='valid')
        ax_curv.plot(curiosity, color='#443', linewidth=0.5, alpha=0.4)
        ax_curv.plot(smoothed_c, color='#ffb74d', linewidth=1.5)
        ax_curv.fill_between(range(len(smoothed_c)), 0, smoothed_c, 
                             alpha=0.15, color='#ffb74d')
    ax_curv.tick_params(colors='#666')
    ax_curv.spines['bottom'].set_color('#333')
    ax_curv.spines['left'].set_color('#333')
    ax_curv.spines['top'].set_visible(False)
    ax_curv.spines['right'].set_visible(False)

    # ── 6. LEARNING CURVE (weight change) ─────────────────────
    ax_learn = fig.add_subplot(gs[2, 2])
    ax_learn.set_facecolor('#1a1a1a')
    ax_learn.set_title('Synaptic learning (Δ weights)', color='#e0e0e0', fontsize=10)
    if log:
        wchanges = [e['weight_change'] for e in log]
        n = len(wchanges)
        window = max(1, n // 50)
        smoothed_w = np.convolve(wchanges, np.ones(window)/window, mode='valid')
        ax_learn.plot(wchanges, color='#333', linewidth=0.5, alpha=0.5)
        ax_learn.plot(smoothed_w, color='#ce93d8', linewidth=1.5)
    ax_learn.tick_params(colors='#666')
    ax_learn.spines['bottom'].set_color('#333')
    ax_learn.spines['left'].set_color('#333')
    ax_learn.spines['top'].set_visible(False)
    ax_learn.spines['right'].set_visible(False)

    # ── 7. ACTIONS DISTRIBUTION ───────────────────────────────
    ax_act = fig.add_subplot(gs[2, 3])
    ax_act.set_facecolor('#1a1a1a')
    ax_act.set_title('Action distribution', color='#e0e0e0', fontsize=10)
    if log:
        actions = [e['action'] for e in log]
        labels = ['LEFT', 'FORWARD', 'RIGHT']
        counts = [actions.count(a) for a in labels]
        colors = ['#ef9a9a', '#80cbc4', '#90caf9']
        bars = ax_act.bar(labels, counts, color=colors, edgecolor='#333', linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax_act.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(count), ha='center', va='bottom', color='#ccc', fontsize=8)
    ax_act.tick_params(colors='#666')
    ax_act.spines['bottom'].set_color('#333')
    ax_act.spines['left'].set_color('#333')
    ax_act.spines['top'].set_visible(False)
    ax_act.spines['right'].set_visible(False)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
    print(f"Report saved: {output_path}")
    return output_path


if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    generate_report()
