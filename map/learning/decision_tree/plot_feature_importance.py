import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from map.data.ee_feature_list import build_feature_label_map, label_feature


def plot_top_n(importances_json, features_csv, n=20, output_png=None):
    with open(importances_json, 'r') as f:
        items = json.load(f)

    df = pd.DataFrame(items)
    df = df.sort_values('importance', ascending=False).head(n)

    mapping = build_feature_label_map(features_csv)

    raw = df['feature'].tolist()
    labels = [mapping.get(x) if x in mapping else label_feature(x) for x in raw]
    values = df['importance'].tolist()

    fig, ax = plt.subplots(figsize=(10, 8))
    labels_plot = list(reversed(labels))
    values_plot = list(reversed(values))
    y = list(range(len(values_plot)))
    ax.barh(y, values_plot, color='#3b82f6')
    ax.set_yticks(y)
    ax.set_yticklabels(labels_plot)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {n} Features (Random Forest Benchmark Model)')
    fig.tight_layout()

    if output_png is None:
        base = os.path.splitext(os.path.basename(importances_json))[0]
        output_png = os.path.join(os.path.dirname(importances_json), f'{base}_top{n}.png')

    fig.savefig(output_png, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    run_plot = True

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    metrics_dir_ = os.path.join(root_, 'training', 'metrics', 'feature_importance')
    features_csv_ = os.path.join(root_, 'training', 'current_features.csv')

    if run_plot:
        cand = [f for f in os.listdir(metrics_dir_) if f.endswith('.json')]
        cand = sorted(cand)
        if len(cand) == 0:
            raise RuntimeError('No feature importance JSON found in metrics directory')
        latest = cand[-1]
        imp_path_ = os.path.join(metrics_dir_, latest)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        n_ = 40
        out_png_ = os.path.join(metrics_dir_, f'{os.path.splitext(latest)[0]}_{ts_}_top{n_}.png')
        plot_top_n(imp_path_, features_csv_, n_, out_png_)

# ========================= EOF ====================================================================
