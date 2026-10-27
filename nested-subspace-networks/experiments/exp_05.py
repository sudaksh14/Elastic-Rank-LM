"""
Experiment 05: End-to-End LLM Low-Rank Performance Pipeline
============================================================

Assembles evaluation results from multiple LLM models (Pythia-2.8B, GPT-Neo-2.7B,
Qwen2-0.5B, Gemma-2B) trained with epoch_curriculum low-rank strategy, and generates
a 1x4 performance-across-ranks figure showing accuracy vs matrix rank with FLOPs
savings on the secondary axis.

Usage:
    # Full assembly + plot from JSON results:
    uv run python experiments/exp_05.py --results_dir results/exp_05/

    # Regenerate plot from existing CSV:
    uv run python experiments/exp_05.py --results_dir results/exp_05/ --plot_only

Outputs (saved to results/exp_05/):
    - llm_performance.csv:                 Assembled DataFrame
    - performance_across_llms_flops.pdf:   1x4 figure
"""

import sys
import os
import argparse
import json
import re
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_05')
PARAMS_JSON = os.path.join(os.path.dirname(__file__), '..', 'data', 'params_with_percentages.json')

MODEL_NAME_MAP = {
    'pythia': 'EleutherAI/pythia-2.8b',
    'gpt-neo2': 'EleutherAI/gpt-neo-2.7B',
    'qwen': 'Qwen/Qwen2-0.5B',
    'gemma': 'google/gemma-2b',
}

MODEL_DISPLAY_NAMES = {
    'EleutherAI/pythia-2.8b': 'Pythia-2.8B',
    'EleutherAI/gpt-neo-2.7B': 'GPT-Neo-2.7B',
    'Qwen/Qwen2-0.5B': 'Qwen2-0.5B',
    'google/gemma-2b': 'Gemma-2B',
}

PLOT_COLORS = ['#007ACC', '#D65F00', '#009E73', '#CC79A7']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_files(results_dir):
    """Load all results_*.json files from a directory into a dict keyed by stem."""
    results_dict = {}
    pattern = os.path.join(results_dir, "results_*.json")
    for file_path in sorted(glob.glob(pattern)):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, 'r') as f:
                results_dict[filename] = json.load(f)
            print(f"  Loaded: {filename}")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    return results_dict


def load_and_assemble(results_dir, params_json):
    """Load JSON results + params metadata, build and return a DataFrame."""
    print("Loading parameter metadata...")
    with open(params_json) as f:
        new_params = json.load(f)

    print(f"Loading results from {results_dir}...")
    results_llms = load_results_files(results_dir)

    if not results_llms:
        print("ERROR: No results_*.json files found.")
        return None

    data_rows = []
    for result_key, result_data in results_llms.items():
        parts = result_key.replace('results_', '').split('_')
        model_short_name = parts[0]
        training_type = '_'.join(parts[1:])
        model_full_name = MODEL_NAME_MAP.get(model_short_name)

        if not model_full_name or not result_data.get('mismatched'):
            continue

        model_params = new_params.get(model_full_name, {})
        total_params = model_params.get('total_params')
        mismatched_data = result_data['mismatched']

        if training_type == 'base':
            row = {
                'model_name': model_full_name,
                'training_type': training_type,
                'rank': np.nan,
                'mismatched_accuracy': mismatched_data.get(
                    'eval_mismatched/rank_baseline/accuracy', np.nan),
                'mismatched_loss': mismatched_data.get(
                    'eval_mismatched/rank_baseline/loss', np.nan),
                'total_params': total_params,
                'rank_params': total_params,
                'params_percentage': '100%',
            }
            data_rows.append(row)

        elif training_type == 'epoch_curriculum':
            ranks = set()
            for key in mismatched_data.keys():
                match = re.search(r'_rank_(\d+)/', key)
                if match:
                    ranks.add(int(match.group(1)))

            rank_params_info = model_params.get('rank_params', {})
            rank_percentage_info = model_params.get('rank_params_percentage', {})

            for rank in sorted(ranks):
                row = {
                    'model_name': model_full_name,
                    'training_type': training_type,
                    'rank': rank,
                    'mismatched_accuracy': mismatched_data.get(
                        f'eval_mismatched_rank_{rank}/accuracy', np.nan),
                    'mismatched_loss': mismatched_data.get(
                        f'eval_mismatched_rank_{rank}/loss', np.nan),
                    'total_params': total_params,
                    'rank_params': rank_params_info.get(str(rank),
                                   rank_params_info.get(rank, np.nan)),
                    'params_percentage': rank_percentage_info.get(str(rank),
                                         rank_percentage_info.get(rank, np.nan)),
                }
                data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df = df[[
        'model_name', 'training_type', 'rank',
        'mismatched_accuracy', 'mismatched_loss',
        'total_params', 'rank_params', 'params_percentage',
    ]]

    def _pct_to_float(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).rstrip('%'))

    df['params_percentage_numeric'] = df['params_percentage'].apply(_pct_to_float)

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_performance(df, output_dir):
    """Generate the 1x4 performance-across-ranks figure."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,
        'axes.linewidth': 1,
        'axes.edgecolor': '#666666',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    model_names = df['model_name'].unique()
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
    fig.patch.set_facecolor('white')

    if n_models == 1:
        axes = [axes]

    for i, model_name in enumerate(model_names):
        ax = axes[i]
        model_df = df[df['model_name'] == model_name]
        curriculum_df = model_df[
            model_df['training_type'] == 'epoch_curriculum'
        ].sort_values('rank')

        base_rows = model_df[model_df['training_type'] == 'base']
        baseline_acc = base_rows['mismatched_accuracy'].iloc[0] if len(base_rows) > 0 else 0.0

        color = PLOT_COLORS[i % len(PLOT_COLORS)]

        ax.plot(curriculum_df['rank'], curriculum_df['mismatched_accuracy'],
                color=color, linewidth=2.5, zorder=3)
        ax.plot(curriculum_df['rank'], curriculum_df['mismatched_accuracy'],
                marker='o', markersize=10, markerfacecolor=color,
                markeredgecolor='white', markeredgewidth=3,
                linestyle='None', zorder=4)
        ax.fill_between(curriculum_df['rank'], curriculum_df['mismatched_accuracy'],
                        color=color, alpha=0.2, zorder=2)

        max_rank_rows = curriculum_df.loc[curriculum_df['rank'] == curriculum_df['rank'].max()]
        max_rank_acc = max_rank_rows['mismatched_accuracy'].iloc[0] if len(max_rank_rows) > 0 else 0.0
        line_label = 'Baseline' if i == 0 else ""
        ax.axhline(y=max(baseline_acc, max_rank_acc), color='#555555',
                   linestyle='--', linewidth=1.5, label=line_label, zorder=4)

        model_text = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.text(0.97, 0.95, model_text,
                transform=ax.transAxes, fontsize=18, fontweight='bold',
                color='white', va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                          edgecolor=color, alpha=0.4),
                zorder=5)

        ax.set_xlabel("Matrix Rank", fontsize=18)
        ax.set_xscale('log', base=2)
        ax.invert_xaxis()
        ax.set_xticks(curriculum_df['rank'].values)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.set_ylim(0.25, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=45, labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.5)

        ax2 = ax.twiny()
        param_percentages = curriculum_df['params_percentage_numeric']
        ax2.set_xscale('log', base=2)
        ax2.set_xlim(ax.get_xlim())
        tick_positions = curriculum_df['rank'].values
        tick_labels = [f'{p:.0f}%' for p in param_percentages]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='left')
        ax2.set_xlabel('FLOps Savings (%)', fontsize=18)
        ax2.spines['right'].set_visible(False)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Accuracy (%)', fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0), ncol=1, fontsize=18, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "performance_across_llms_flops.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                        help='Directory containing results_*.json files')
    parser.add_argument('--params_json', type=str, default=PARAMS_JSON,
                        help='Path to params_with_percentages.json')
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip JSON loading, regenerate plot from CSV')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "llm_performance.csv")

    if args.plot_only:
        if not os.path.exists(csv_path):
            print(f"No CSV found at {csv_path}. Run without --plot_only first.")
            sys.exit(1)
        print(f"Loading DataFrame from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        df = load_and_assemble(results_dir, os.path.abspath(args.params_json))
        if df is None or df.empty:
            print("No data assembled. Exiting.")
            sys.exit(1)
        df.to_csv(csv_path, index=False)
        print(f"DataFrame saved to {csv_path}  ({len(df)} rows)")

    print(f"\n{df.to_string()}\n")
    plot_performance(df, results_dir)

    print(f"\nDone. Results are in {results_dir}/")
    print(f"  - llm_performance.csv")
    print(f"  - performance_across_llms_flops.pdf")
