#!/usr/bin/env python3
"""
Train small sklearn models and export ONNX for in-browser demos (ONNX Runtime Web).

Run from repo root (requires: pip install scikit-learn skl2onnx onnx numpy):
    python scripts/export_tutorial_onnx.py

Outputs go to app/onnx_models/ — commit those files so GitHub Pages can serve them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "onnx_models"
OUT.mkdir(parents=True, exist_ok=True)

try:
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris, make_classification, make_regression
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.naive_bayes import GaussianNB
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError as e:
    print("Install dependencies: pip install scikit-learn skl2onnx onnx numpy", file=sys.stderr)
    raise e


def save_onnx(model, path: Path, initial_types, options=None):
    opts = options or {}
    onx = convert_sklearn(
        model,
        initial_types=initial_types,
        options=opts,
        target_opset=12,
    )
    path.write_bytes(onx.SerializeToString())
    print(f"Wrote {path} ({path.stat().st_size // 1024} KB)")


def main():
    # --- Iris K-Means (petal length & width) ---
    iris = load_iris()
    X_iris = iris.data[:, 2:4].astype(np.float32)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_iris)
    save_onnx(
        km,
        OUT / "kmeans.onnx",
        [("float_input", FloatTensorType([None, 2]))],
    )

    # --- Drug-style Naive Bayes (5 numeric features, 5 classes) ---
    rng = np.random.default_rng(42)
    X_drug = rng.normal(size=(800, 5)).astype(np.float32)
    y_drug = rng.integers(0, 5, size=800)
    gnb = GaussianNB()
    gnb.fit(X_drug, y_drug)
    save_onnx(
        gnb,
        OUT / "naive_bayes.onnx",
        [("float_input", FloatTensorType([None, 5]))],
    )

    # --- Crime-style classification (15 features, normalized like the tutorial UI) ---
    X_MEANS_C = np.array(
        [
            18435,
            14.835,
            115.36,
            0.1966,
            1148.9,
            11.260,
            23.228,
            36.587,
            11.047,
            2021.0,
            41.844,
            -87.670,
            12.932,
            2.972,
            6.619,
        ],
        dtype=np.float32,
    )
    X_STD_C = np.array(
        [
            11375,
            12.067,
            59.395,
            0.39753,
            702.12,
            7.016,
            13.978,
            21.535,
            5.8643,
            2.0642,
            0.0879,
            0.0592,
            6.0050,
            2.0050,
            3.3636,
        ],
        dtype=np.float32,
    )
    X_raw_c, y_c = make_classification(
        n_samples=3000,
        n_features=15,
        n_informative=12,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X_raw_c = X_raw_c.astype(np.float32)
    # Shift/scale synthetic draws toward the tutorial's feature ranges
    X_raw_c = X_raw_c * (X_STD_C * 0.25) + X_MEANS_C
    Xn_c = (X_raw_c - X_MEANS_C) / X_STD_C

    mlp_shallow = MLPClassifier(
        hidden_layer_sizes=(32,),
        max_iter=500,
        random_state=42,
    )
    mlp_shallow.fit(Xn_c, y_c)
    save_onnx(
        mlp_shallow,
        OUT / "MLP_Model_Classification.onnx",
        [("float_input", FloatTensorType([None, 15]))],
        options={id(mlp_shallow): {"zipmap": False}},
    )

    mlp_deep = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        max_iter=500,
        random_state=43,
    )
    mlp_deep.fit(Xn_c, y_c)
    save_onnx(
        mlp_deep,
        OUT / "DL_Model_Classification.onnx",
        [("float_input", FloatTensorType([None, 15]))],
        options={id(mlp_deep): {"zipmap": False}},
    )

    # --- Regression (55 features): same mean/std layout as supervised_learning.html ---
    ward_means = np.array([0.0179, 0.0187, 0.0280, 0.0267, 0.0243], dtype=np.float32)
    X_MEANS_R = np.concatenate(
        [
            np.array([2.9716, 6.6190, 2021.0, 41.844, -87.670], dtype=np.float32),
            ward_means,
            np.zeros(45, dtype=np.float32),
        ]
    )
    ward_std = np.array([0.1329, 0.1357, 0.1653, 0.1614, 0.1543], dtype=np.float32)
    X_STD_R = np.concatenate(
        [
            np.array([2.0050, 3.3636, 2.0609, 0.0879, 0.0592], dtype=np.float32),
            ward_std,
            np.ones(45, dtype=np.float32),
        ]
    )
    X_raw_r, y_r = make_regression(
        n_samples=2500,
        n_features=55,
        n_informative=20,
        noise=15.0,
        random_state=42,
    )
    X_raw_r = X_raw_r.astype(np.float32)
    X_raw_r = X_raw_r * (X_STD_R * 0.2) + X_MEANS_R
    Xn_r = (X_raw_r - X_MEANS_R) / X_STD_R
    y_r = np.clip(np.abs(y_r) * 30 + 400, 200, 5000).astype(np.float32)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xn_r, y_r)
    save_onnx(
        ridge,
        OUT / "linreg_model_Regression.onnx",
        [("float_input", FloatTensorType([None, 55]))],
    )

    mlp_r = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        max_iter=500,
        random_state=42,
    )
    mlp_r.fit(Xn_r, y_r)
    save_onnx(
        mlp_r,
        OUT / "mlp_model_Regression.onnx",
        [("float_input", FloatTensorType([None, 55]))],
    )

    dl_r = MLPRegressor(
        hidden_layer_sizes=(96, 48, 24),
        max_iter=500,
        random_state=43,
    )
    dl_r.fit(Xn_r, y_r)
    save_onnx(
        dl_r,
        OUT / "dl_model_Regression.onnx",
        [("float_input", FloatTensorType([None, 55]))],
    )

    mix_r = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=44,
    )
    mix_r.fit(Xn_r, y_r)
    save_onnx(
        mix_r,
        OUT / "mix_model_Regression.onnx",
        [("float_input", FloatTensorType([None, 55]))],
    )

    print("Done. Commit app/onnx_models/*.onnx and deploy.")


if __name__ == "__main__":
    main()
