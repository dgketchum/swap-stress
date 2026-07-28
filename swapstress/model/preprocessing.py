"""
Preprocessing for the direct suction prediction task.

Provides imputation, standardization, and categorical encoding shared
between RF and NN trainers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Categorical features: integer-coded class columns that should get
# embeddings (NN) or one-hot encoding rather than be treated as continuous.
CATEGORICAL_FEATURES = frozenset(
    {
        "c3s_lccs_class_mode",
        "glc10_lc",
        "nlcd",
        "cdl_cultivated_mode",
        "cdl_crop_mode",
        "cdl_simple_crop_mode",
        "us_lith",
    }
)


def build_preprocessor(add_indicator: bool = True) -> SimpleImputer:
    """Create median imputer for handling NaN values in features."""
    return SimpleImputer(strategy="median", add_indicator=add_indicator)


def get_categorical_feature_columns(feature_cols: list[str]) -> list[str]:
    """Return the subset of *feature_cols* that are categorical."""
    return sorted(c for c in feature_cols if c in CATEGORICAL_FEATURES)


def get_numeric_feature_columns(feature_cols: list[str]) -> list[str]:
    """Return the subset of *feature_cols* that are numeric (continuous)."""
    return [c for c in feature_cols if c not in CATEGORICAL_FEATURES]


def prepare_rf_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_features: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimpleImputer]:
    """Build imputed X/y arrays for Random Forest.

    Returns
    -------
    X_train, X_test, y_train, y_test, imputer
    """
    imputer = build_preprocessor(add_indicator=False)
    X_train = imputer.fit_transform(train_df[all_features].values)
    X_test = imputer.transform(test_df[all_features].values)
    y_train = train_df["log10_suction_cm"].values
    y_test = test_df["log10_suction_cm"].values
    return X_train, X_test, y_train, y_test, imputer


class NNPreprocessor:
    """Preprocessing pipeline for NN models.

    Handles:
    - Median imputation (fit on train)
    - Standard scaling of numeric features (fit on train)
    - Categorical encoding with unknown/missing bucket (fit on train)
    """

    def __init__(self):
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.category_maps: Dict[str, Dict] = {}
        self.cat_cardinalities: list[int] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        all_features: list[str],
    ) -> "NNPreprocessor":
        """Fit imputer, scaler, and category maps on training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data.
        all_features : list of str
            Feature columns including theta.

        Notes
        -----
        Categorical columns are **not** median-imputed.  Missing values map
        to the reserved unknown bucket (index 0).  Only numeric columns go
        through the median imputer and standard scaler.
        """
        # Separate numeric and categorical
        self.cat_cols = [c for c in all_features if c in CATEGORICAL_FEATURES]
        self.num_cols = [c for c in all_features if c not in CATEGORICAL_FEATURES]

        # Build category maps from raw (pre-imputation) training data.
        # 0 is reserved for unknown / missing.
        self.category_maps = {}
        self.cat_cardinalities = []
        for col in self.cat_cols:
            raw = train_df[col].dropna()
            sorted_vals = sorted(set(int(v) for v in raw.unique()))
            mapping = {v: i + 1 for i, v in enumerate(sorted_vals)}
            self.category_maps[col] = mapping
            self.cat_cardinalities.append(len(mapping) + 1)  # +1 for unknown bucket

        # Fit median imputer on numeric columns only
        self.imputer = SimpleImputer(strategy="median", add_indicator=False)
        self.imputer.fit(train_df[self.num_cols].values)

        # Fit scaler on imputed numeric columns
        num_imputed = self.imputer.transform(train_df[self.num_cols].values)
        self.scaler = StandardScaler()
        self.scaler.fit(num_imputed)

        return self

    def transform(
        self,
        df: pd.DataFrame,
        all_features: list[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform a dataframe to (X_num, X_cat, y).

        Returns
        -------
        X_num : np.ndarray, shape (n, n_numeric)
            Imputed and standardized numeric features.
        X_cat : np.ndarray of int64, shape (n, n_categorical)
            Integer-encoded categorical features (0 = unknown/missing).
        y : np.ndarray
            Target values.
        """
        # Numeric: impute then scale
        X_num = self.scaler.transform(
            self.imputer.transform(df[self.num_cols].values)
        ).astype(np.float32)

        # Categorical: map raw values to indices; NaN -> 0 (unknown bucket)
        if self.cat_cols:
            X_cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
            for j, col in enumerate(self.cat_cols):
                mapping = self.category_maps[col]
                raw = df[col].values
                for i, v in enumerate(raw):
                    if pd.isna(v):
                        X_cat[i, j] = 0  # unknown bucket
                    else:
                        X_cat[i, j] = mapping.get(int(v), 0)
        else:
            X_cat = np.zeros((len(df), 0), dtype=np.int64)

        y = df["log10_suction_cm"].values.astype(np.float32)
        return X_num, X_cat, y

    def transform_flat(
        self,
        df: pd.DataFrame,
        all_features: list[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform to a single flat array (for VanillaMLP with one-hot cats).

        Returns
        -------
        X : np.ndarray of float32, shape (n, n_numeric + sum(cardinalities))
        y : np.ndarray
        """
        X_num, X_cat, y = self.transform(df, all_features)

        if X_cat.shape[1] == 0:
            return X_num, y

        # One-hot encode categorical columns
        one_hot_parts = []
        for j, card in enumerate(self.cat_cardinalities):
            oh = np.zeros((len(df), card), dtype=np.float32)
            oh[np.arange(len(df)), X_cat[:, j]] = 1.0
            one_hot_parts.append(oh)

        X = np.concatenate([X_num] + one_hot_parts, axis=1)
        return X, y

    def save(self, output_dir: str) -> None:
        """Save preprocessor artifacts."""
        import json
        import os
        import joblib

        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(
            {"imputer": self.imputer, "scaler": self.scaler},
            os.path.join(output_dir, "direct_nn_preprocessor.joblib"),
        )
        with open(os.path.join(output_dir, "direct_nn_category_maps.json"), "w") as f:
            json.dump(
                {
                    "cat_cols": self.cat_cols,
                    "num_cols": self.num_cols,
                    "category_maps": {
                        k: {str(kk): vv for kk, vv in v.items()}
                        for k, v in self.category_maps.items()
                    },
                    "cat_cardinalities": self.cat_cardinalities,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, output_dir: str) -> "NNPreprocessor":
        """Load preprocessor artifacts."""
        import json
        import os
        import joblib

        obj = cls()
        saved = joblib.load(os.path.join(output_dir, "direct_nn_preprocessor.joblib"))
        obj.imputer = saved["imputer"]
        obj.scaler = saved["scaler"]

        with open(os.path.join(output_dir, "direct_nn_category_maps.json")) as f:
            meta = json.load(f)
        obj.cat_cols = meta["cat_cols"]
        obj.num_cols = meta["num_cols"]
        obj.category_maps = {
            k: {int(kk): vv for kk, vv in v.items()}
            for k, v in meta["category_maps"].items()
        }
        obj.cat_cardinalities = meta["cat_cardinalities"]
        return obj
