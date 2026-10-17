"""Plot existing JSONL metrics without loading models or changing a training run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path('networks/diffuscene/runs/livingroom_uncond_100k'))
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    groups = {}
    for event in ('train', 'validation'):
        by_step = {r['step']: r for r in records if r['event'] == event}
        groups[event] = [by_step[step] for step in sorted(by_step)]
    train, val = groups['train'], groups['validation']
    tx, ty = np.array([r['step'] for r in train]), np.array([r['loss'] for r in train])
    vx, vy = np.array([r['step'] for r in val]), np.array([r['loss'] for r in val])
    assert np.isfinite(ty).all() and np.isfinite(vy).all()
    window = 50
    smooth = np.convolve(ty, np.ones(window) / window, mode='valid')
    sx = tx[window - 1:]
    best = min(val, key=lambda r: r['loss'])
    total_steps = int(tx[-1])
    late_start = 100000 if total_steps >= 500000 else min(10000, total_steps // 10)
    lr_changes = [train[i - 1]['step'] for i in range(1, len(train))
                  if train[i].get('lr') != train[i - 1].get('lr')]
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), constrained_layout=True)
    for ax in axes:
        ax.plot(tx, ty, color='#2679bf', alpha=.16, linewidth=.7, label='Train: logged interval mean')
        ax.plot(sx, smooth, color='#2679bf', linewidth=1.8, label='Train: trailing ~1k-step mean')
        ax.plot(vx, vy, color='#d85b30', linewidth=1.6, label='Validation: full split')
        ax.scatter([best['step']], [best['loss']], color='#d85b30', marker='*', s=130, zorder=5)
        for i, step in enumerate(lr_changes):
            ax.axvline(step, color='#777777', linestyle='--', linewidth=.8, alpha=.4,
                       label='Learning-rate decay' if i == 0 else None)
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel('Weighted diffusion loss')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value / 1000:g}k'))
        ax.grid(alpha=.18)
    axes[0].set_title('Full training (log loss scale)')
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, tx[-1])
    axes[0].legend(fontsize=9, loc='upper right')
    axes[1].set_title(f'After {late_start / 1000:g}k steps (linear loss scale)')
    axes[1].set_xlim(late_start, tx[-1])
    late_values = np.concatenate([ty[tx >= late_start], vy[vx >= late_start]])
    lo, hi = float(late_values.min()), float(late_values.max())
    axes[1].set_ylim(lo - .1 * (hi - lo), hi + .1 * (hi - lo))
    axes[1].text(.98, .98, f"Best val: {best['loss']:.5f} at {best['step'] / 1000:g}k\n"
                 f"Last val: {val[-1]['loss']:.5f}\nLast train interval: {train[-1]['loss']:.5f}",
                 transform=axes[1].transAxes, va='top', ha='right', fontsize=10,
                 bbox={'facecolor': 'white', 'edgecolor': '#dddddd', 'alpha': .9})
    fig.suptitle(f'DiffuScene | Livingroom | Unconditional | {total_steps:,} steps', fontsize=15)
    fig.savefig(args.run / 'loss_curves.png', dpi=180)
    fig.savefig(args.run / 'loss_curves.pdf')
    plt.close(fig)

    summary = {'train_log_points': len(train), 'validation_points': len(val),
               'initial_train_loss': train[0]['loss'], 'initial_val_loss': val[0]['loss'],
               'last_train_interval_loss': train[-1]['loss'], 'last_val_loss': val[-1]['loss'],
               'best_val_loss': best['loss'], 'best_val_step': best['step'],
               'last_vs_best_val_change_percent': 100 * (val[-1]['loss'] / best['loss'] - 1),
               'lr_decay_steps': lr_changes,
               'milestones': [], 'late_windows': [],
               'loss_definition': 'Mean over samples of alpha_bar(t) * (MSE over 8 geometry channels + MSE over 33 class channels). Includes empty slots; IoU loss disabled.',
               'train_curve': 'Logged interval sample mean; smoothing is a trailing mean of 50 logged points (approximately 1000 steps).',
               'validation_curve': 'Full 810-scene validation split every 1000 steps with fixed validation RNG seed.'}
    train_map, val_map = {r['step']: r for r in train}, {r['step']: r for r in val}
    milestones = {1000, 5000, 10000, 20000, 50000, 75000, 100000,
                  best['step'], total_steps, *lr_changes}
    for step in sorted(milestones):
        if step in train_map and step in val_map:
            summary['milestones'].append({'step': step, 'train_interval_loss': train_map[step]['loss'],
                                          'val_loss': val_map[step]['loss']})
    window_ends = sorted({min(50000, total_steps), best['step'], total_steps})
    for end in window_ends:
        start = max(0, end - 10000)
        summary['late_windows'].append({'steps': [start, end],
                                       'mean_train_logged_loss': float(ty[(tx > start) & (tx <= end)].mean()),
                                       'mean_val_loss': float(vy[(vx > start) & (vx <= end)].mean())})
    summary['final_validation_components_unweighted'] = {
        key: val[-1][key] for key in ('loss.trans', 'loss.size', 'loss.angle', 'loss.class')}
    (args.run / 'loss_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
