"""
Train tabular neural networks for direct suction prediction.

Supports three architectures via --model-name:
    mlp             — VanillaMLP (flat input, one-hot categoricals)
    mlp_embeddings  — MLPWithEmbeddings (learned categorical embeddings)
    ft_transformer  — FTTransformer (feature tokenizer + transformer)

Uses the same observation table, feature groups, and spatial holdout as the
RF trainer so results are directly comparable.

Usage:
    python -m map.learning.tabular_nn.train_direct_nn \\
        --config configs/train_9km_conus_mlp.toml

    python -m map.learning.tabular_nn.train_direct_nn \\
        --obs-table ... --output-dir ... --model-name mlp
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from map.learning.direct.data import (
    prepare_direct_data,
    write_split_manifest,
)
from map.learning.direct.preprocessing import NNPreprocessor
from map.learning.direct.reporting import evaluate_and_report
from map.learning.tabular_nn.dataset import FlatDataset, SplitDataset
from map.learning.tabular_nn.lightning_module import DirectRegressionModule
from map.learning.tabular_nn.models import (
    FTTransformer,
    MLPWithEmbeddings,
    VanillaMLP,
)

MODEL_REGISTRY = {
    "mlp": VanillaMLP,
    "mlp_embeddings": MLPWithEmbeddings,
    "ft_transformer": FTTransformer,
}


def train_and_evaluate(
    obs_table_path: str,
    output_dir: str,
    model_name: str = "mlp",
    exclude_groups: Optional[list[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    resolution_m: float = 250,
    split_manifest: str | None = None,
    # Training hyperparameters
    batch_size: int = 1024,
    max_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    # Architecture hyperparameters
    hidden_dim: int = 256,
    num_hidden_layers: int = 3,
    dropout: float = 0.2,
    embedding_dim: int = 16,
    d_token: int = 192,
    n_blocks: int = 3,
    n_heads: int = 8,
    attn_dropout: float = 0.2,
    ff_dropout: float = 0.1,
    # Config/provenance
    config_dict: Optional[Dict] = None,
) -> Dict:
    """Train an NN model and evaluate on held-out test set.

    The workflow:
    1. Load data and build train/val/test spatial split
    2. Fit preprocessor on train data
    3. Train model with early stopping on val RMSE
    4. Refit on train+val with the best epoch count
    5. Evaluate on test set
    6. Write standard artifacts

    Returns
    -------
    dict
        Results dictionary (same schema as RF trainer).
    """
    os.makedirs(output_dir, exist_ok=True)
    L.seed_everything(random_state, workers=True)

    is_split = model_name in ("mlp_embeddings", "ft_transformer")

    # ------------------------------------------------------------------
    # 1. Data loading with train/val/test split
    # ------------------------------------------------------------------
    data = prepare_direct_data(
        obs_table_path=obs_table_path,
        output_dir=output_dir,
        exclude_groups=exclude_groups,
        drop_blocking_features=True,
        resolution_m=resolution_m,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        split_manifest=split_manifest,
    )

    all_features = data["all_features"]
    train_df = data["train_df"]
    test_df = data["test_df"]
    train_sites = data["train_sites"]
    test_sites = data["test_sites"]

    # NN requires a validation split
    if "val_df" not in data or data["val_df"] is None:
        raise ValueError(
            "NN training requires a validation split. Either pass val_size > 0 "
            "or use a split manifest that includes val_groups."
        )
    val_df = data["val_df"]
    val_sites = data["val_sites"]

    # Write split manifest to the shared path (config-specified or output_dir)
    manifest_path = split_manifest or os.path.join(output_dir, "spatial_split.json")
    if not os.path.exists(manifest_path):
        write_split_manifest(
            manifest_path,
            train_groups=train_sites,
            test_groups=test_sites,
            val_groups=val_sites,
            random_state=random_state,
            resolution_m=resolution_m,
        )

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    preprocessor = NNPreprocessor().fit(train_df, all_features)

    n_num = len(preprocessor.num_cols)
    n_cat = len(preprocessor.cat_cols)
    cat_cards = preprocessor.cat_cardinalities

    if is_split:
        X_train_num, X_train_cat, y_train = preprocessor.transform(
            train_df, all_features
        )
        X_val_num, X_val_cat, y_val = preprocessor.transform(val_df, all_features)
        train_ds = SplitDataset(X_train_num, X_train_cat, y_train)
        val_ds = SplitDataset(X_val_num, X_val_cat, y_val)
    else:
        X_train_flat, y_train = preprocessor.transform_flat(train_df, all_features)
        X_val_flat, y_val = preprocessor.transform_flat(val_df, all_features)
        train_ds = FlatDataset(X_train_flat, y_train)
        val_ds = FlatDataset(X_val_flat, y_val)
        n_flat = X_train_flat.shape[1]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4,
    )

    # ------------------------------------------------------------------
    # 3. Build model
    # ------------------------------------------------------------------
    if model_name == "mlp":
        model = VanillaMLP(
            n_features=n_flat,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    elif model_name == "mlp_embeddings":
        model = MLPWithEmbeddings(
            n_num_features=n_num,
            cat_cardinalities=cat_cards,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    elif model_name == "ft_transformer":
        model = FTTransformer(
            n_num_features=n_num,
            cat_cardinalities=cat_cards,
            d_token=d_token,
            n_blocks=n_blocks,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    lit_module = DirectRegressionModule(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        split_input=is_split,
    )

    # ------------------------------------------------------------------
    # 4. Train with early stopping
    # ------------------------------------------------------------------
    device = "gpu" if torch.cuda.is_available() else "cpu"

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-{epoch:03d}-{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=patience,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        default_root_dir=output_dir,
        enable_progress_bar=True,
        deterministic=True,
    )
    trainer.fit(lit_module, train_loader, val_loader)

    best_val_rmse_score = checkpoint_cb.best_model_score
    stopped_epoch = trainer.current_epoch
    best_val_rmse = (
        float(best_val_rmse_score.cpu().item())
        if best_val_rmse_score is not None
        else None
    )

    # Extract best epoch from checkpoint filename (pattern: best-{epoch:03d}-...)
    best_ckpt_path = checkpoint_cb.best_model_path
    import re

    _epoch_match = re.search(r"best-(\d+)-", os.path.basename(best_ckpt_path or ""))
    best_epoch_num = int(_epoch_match.group(1)) if _epoch_match else stopped_epoch
    # Lightning epochs are 0-indexed; refit needs epoch *count*
    refit_epochs = best_epoch_num + 1

    print(
        f"\nBest val RMSE: {best_val_rmse:.4f} at epoch {best_epoch_num} "
        f"(stopped at epoch {stopped_epoch})"
    )

    # Capture training history
    training_history = {
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch_num,
        "stopped_epoch": stopped_epoch,
        "best_checkpoint": best_ckpt_path,
    }

    # ------------------------------------------------------------------
    # 5. Refit on train+val with best epoch count
    # ------------------------------------------------------------------
    print(f"\nRefitting on train+val for {refit_epochs} epochs (best epoch)...")

    # Merge train and val for refit
    import pandas as pd

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    if is_split:
        X_tv_num, X_tv_cat, y_tv = preprocessor.transform(trainval_df, all_features)
        trainval_ds = SplitDataset(X_tv_num, X_tv_cat, y_tv)
    else:
        X_tv_flat, y_tv = preprocessor.transform_flat(trainval_df, all_features)
        trainval_ds = FlatDataset(X_tv_flat, y_tv)

    trainval_loader = DataLoader(
        trainval_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    # Fresh model for refit
    if model_name == "mlp":
        refit_model = VanillaMLP(
            n_features=n_flat,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    elif model_name == "mlp_embeddings":
        refit_model = MLPWithEmbeddings(
            n_num_features=n_num,
            cat_cardinalities=cat_cards,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    else:
        refit_model = FTTransformer(
            n_num_features=n_num,
            cat_cardinalities=cat_cards,
            d_token=d_token,
            n_blocks=n_blocks,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    refit_lit = DirectRegressionModule(
        model=refit_model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        split_input=is_split,
    )

    refit_trainer = L.Trainer(
        max_epochs=refit_epochs,
        accelerator=device,
        devices=1,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        deterministic=True,
    )
    refit_trainer.fit(refit_lit, trainval_loader)

    # ------------------------------------------------------------------
    # 6. Test evaluation
    # ------------------------------------------------------------------
    if is_split:
        X_test_num, X_test_cat, y_test = preprocessor.transform(test_df, all_features)
        test_ds = SplitDataset(X_test_num, X_test_cat, y_test)
    else:
        X_test_flat, y_test = preprocessor.transform_flat(test_df, all_features)
        test_ds = FlatDataset(X_test_flat, y_test)

    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4)
    refit_trainer.test(refit_lit, test_loader)

    y_pred = refit_lit.test_predictions
    y_true = refit_lit.test_targets

    # Validation metrics from best checkpoint for reporting
    val_metrics = {"val_rmse": best_val_rmse}

    # ------------------------------------------------------------------
    # 7. Standard reporting
    # ------------------------------------------------------------------
    results = evaluate_and_report(
        y_test=y_true,
        y_pred=y_pred,
        test_df=test_df,
        output_dir=output_dir,
        all_features=all_features,
        train_df=trainval_df,
        train_sites=train_sites | val_sites,
        test_sites=test_sites,
        resolution_m=resolution_m,
        feature_importance=None,
        extra_config={
            "obs_table": obs_table_path,
            "exclude_groups": exclude_groups,
            "model_name": model_name,
            "test_size": test_size,
            "val_size": val_size,
            "random_state": random_state,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "dropout": dropout,
        },
        model_family="nn",
        model_name=model_name,
        validation_metrics=val_metrics,
        training_summary=training_history,
    )

    # ------------------------------------------------------------------
    # 8. Save NN-specific artifacts
    # ------------------------------------------------------------------
    preprocessor.save(output_dir)

    features_path = os.path.join(output_dir, "direct_nn_features.json")
    with open(features_path, "w") as f:
        json.dump(all_features, f, indent=2)

    # Save final model weights
    model_path = os.path.join(output_dir, "direct_nn_model.pt")
    torch.save(refit_model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Training history CSV
    if hasattr(trainer, "callback_metrics"):
        history = {
            "refit_epochs": refit_epochs,
            "best_val_rmse": best_val_rmse,
            "stopped_epoch": stopped_epoch,
        }
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # Provenance
    if config_dict is not None:
        from map.config import input_checksum, write_provenance

        prov_path = write_provenance(
            output_dir=output_dir,
            config=config_dict,
            run_type="train_nn",
            extras={
                "inputs": {
                    "obs_table_sha256": input_checksum(obs_table_path),
                    "obs_table_n_rows": len(data["df"]),
                    "obs_table_n_cols": data["df"].shape[1],
                },
                "outputs": {
                    "model_name": model_name,
                    "n_features": len(all_features),
                    "n_numeric_features": n_num,
                    "n_categorical_features": n_cat,
                    "cat_cardinalities": cat_cards,
                    "n_train": len(train_df),
                    "n_val": len(val_df),
                    "n_test": len(test_df),
                    "n_train_sites": len(train_sites),
                    "n_val_sites": len(val_sites),
                    "n_test_sites": len(test_sites),
                    "best_val_rmse": best_val_rmse,
                    "refit_epochs": refit_epochs,
                },
                "upstream": None,
            },
        )
        print(f"Saved provenance to {prov_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train tabular NN: EE features + theta -> log10(suction_cm)",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to TOML run config."
    )
    parser.add_argument("--obs-table", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="mlp | mlp_embeddings | ft_transformer",
    )
    parser.add_argument("--exclude-groups", type=str, nargs="*", default=None)
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Path to existing spatial_split.json.",
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--d-token", type=int, default=None)
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--attn-dropout", type=float, default=None)
    parser.add_argument("--ff-dropout", type=float, default=None)
    args = parser.parse_args()

    from map.config import feature_groups_to_exclude, load_config

    config = load_config(args.config, vars(args))

    if not config.get("obs_table"):
        parser.error("--obs-table is required (via CLI or TOML config)")
    if not config.get("output_dir"):
        parser.error("--output-dir is required (via CLI or TOML config)")
    if not config.get("model_name"):
        parser.error("--model-name is required (via CLI or TOML config)")

    exclude_groups = config.get("exclude_groups")
    if config.get("feature_groups") is not None:
        exclude_groups = feature_groups_to_exclude(config["feature_groups"])

    train_and_evaluate(
        obs_table_path=config["obs_table"],
        output_dir=config["output_dir"],
        model_name=config["model_name"],
        exclude_groups=exclude_groups,
        test_size=config.get("test_size", 0.2),
        val_size=config.get("val_size", 0.2),
        random_state=config.get("random_state", 42),
        resolution_m=config.get("resolution_m", 250),
        split_manifest=config.get("split_manifest"),
        batch_size=config.get("batch_size", 1024),
        max_epochs=config.get("max_epochs", 200),
        learning_rate=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 1e-5),
        patience=config.get("patience", 20),
        hidden_dim=config.get("hidden_dim", 256),
        num_hidden_layers=config.get("num_hidden_layers", 3),
        dropout=config.get("dropout", 0.2),
        embedding_dim=config.get("embedding_dim", 16),
        d_token=config.get("d_token", 192),
        n_blocks=config.get("n_blocks", 3),
        n_heads=config.get("n_heads", 8),
        attn_dropout=config.get("attn_dropout", 0.2),
        ff_dropout=config.get("ff_dropout", 0.1),
        config_dict=config,
    )
