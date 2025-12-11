import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

from map.data.ee_feature_list import build_feature_label_map, label_feature


def plot_top_n(importances_json, features_csv, n=20, output_png=None,
               base_data=False, include_soils=True):
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
        suffix_parts = []
        if base_data and 'BaseData' not in base:
            suffix_parts.append('BaseData')
        if (not include_soils) and 'NoSoils' not in base:
            suffix_parts.append('NoSoils')
        suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ''
        output_png = os.path.join(os.path.dirname(importances_json), f'{base}{suffix}_top{n}.png')

    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_rf_tree(model_path, features_path=None, tree_index=0,
                 max_depth=4, n_features=20, output_png=None):
    """Plot a single decision tree from a RandomForest model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found: {model_path}')

    model = joblib.load(model_path)
    if not hasattr(model, 'estimators_') or len(model.estimators_) == 0:
        raise ValueError('Loaded model does not contain any estimators_ to plot')

    if tree_index < 0 or tree_index >= len(model.estimators_):
        raise IndexError(f'tree_index {tree_index} is out of range for forest of size {len(model.estimators_)}')

    feature_names = None
    if features_path and os.path.exists(features_path):
        if features_path.endswith('.json'):
            with open(features_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                feature_names = list(data)
            else:
                feature_names = [str(x) for x in data]
        elif features_path.endswith('.csv'):
            df = pd.read_csv(features_path)
            col = 'features' if 'features' in df.columns else df.columns[0]
            feature_names = df[col].dropna().astype(str).tolist()

    tree = model.estimators_[tree_index]

    # Optionally collapse less important features into a single label
    # to reduce visual clutter in the rendered tree.
    if feature_names is not None and hasattr(tree, 'feature_importances_'):
        importances = list(enumerate(tree.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        keep_idx = {i for i, _ in importances[:n_features] if i < len(feature_names)}
        collapsed = []
        for i, name in enumerate(feature_names):
            if i in keep_idx:
                collapsed.append(name)
            else:
                collapsed.append('other')
        feature_names = collapsed

    fig, ax = plt.subplots(figsize=(20, 20))
    plot_tree(
        tree,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f'Random Forest Tree {tree_index}')
    fig.tight_layout()

    if output_png is None:
        base = os.path.splitext(os.path.basename(model_path))[0]
        output_png = os.path.join(
            os.path.dirname(model_path),
            f'{base}_tree{tree_index}_depth{max_depth}_top{n_features}.png',
        )

    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    return output_png


if __name__ == '__main__':
    run_plot = False
    run_tree_plot = True
    base_data_only = True
    include_soils = False

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS', 'soils', 'swapstress')
    metrics_dir_ = os.path.join(root_, 'training', 'metrics', 'feature_importance')
    models_dir_ = os.path.join(root_, 'training', 'models')
    features_csv_ = os.path.join(root_, 'training', 'current_features.csv')

    if run_plot:
        cand = [f for f in os.listdir(metrics_dir_) if f.endswith('.json')]
        selected = []
        for f in cand:
            has_base = 'BaseData' in f
            has_no_soils = 'NoSoils' in f
            if has_base != base_data_only:
                continue
            if has_no_soils != (not include_soils):
                continue
            selected.append(f)
        cand = sorted(selected)
        cand = sorted(cand)
        if len(cand) == 0:
            raise RuntimeError('No feature importance JSON found in metrics directory')
        latest = cand[-1]
        imp_path_ = os.path.join(metrics_dir_, latest)
        ts_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        n_ = 40
        base_name = os.path.splitext(latest)[0]
        suffix_parts = []
        if base_data_only and 'BaseData' not in base_name:
            suffix_parts.append('BaseData')
        if (not include_soils) and 'NoSoils' not in base_name:
            suffix_parts.append('NoSoils')
        suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ''
        out_png_ = os.path.join(metrics_dir_, f'{base_name}_{ts_}{suffix}_top{n_}.png')
        plot_top_n(imp_path_, features_csv_, n_, out_png_,
                   base_data=base_data_only, include_soils=include_soils)

    if run_tree_plot:
        model_path_ = os.path.join(models_dir_, 'rf_gshp_model.joblib')
        features_json_ = os.path.join(models_dir_, 'rf_gshp_features.json')
        ts_tree_ = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_tree_png_ = os.path.join(
            models_dir_,
            f'rf_gshp_tree0_depth4_top20_{ts_tree_}.png',
        )
        try:
            saved_path = plot_rf_tree(
                model_path=model_path_,
                features_path=features_json_,
                tree_index=0,
                max_depth=8,
                n_features=None,
                output_png=out_tree_png_,
            )
            print(f'Saved Random Forest tree plot to {saved_path}')
        except (FileNotFoundError, ValueError, IndexError) as exc:
            print(f'Failed to plot RF tree: {exc}')

# ========================= EOF ====================================================================
