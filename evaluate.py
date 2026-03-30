"""
evaluate.py — Évaluation complète du modèle LSTM Météo
=======================================================
Charge le modèle sauvegardé et génère :
  - Métriques : MSE, RMSE, MAE, R²
  - Graphique prédictions vs réalité (valeurs dé-normalisées)
  - Distribution des résidus + scatter réel/prédit
  - Export JSON du rapport complet

Usage :
    python evaluate.py
    python evaluate.py --model best_lstm_model.keras --output eval_results/
"""

import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.rnn_model import WeatherLSTM  # noqa: F401
from utils.data_loader import load_jena_climate, make_timeseries_datasets
from utils.visualisation import plot_predictions, plot_error_distribution

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_lstm_model.keras"
OUTPUT_DIR = "evaluation_results_lstm"
DATA_DIR   = "data"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation LSTM Météo")
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--data",   default=DATA_DIR)
    args = parser.parse_args()

    ensure_dir(args.output)

    print("=" * 60)
    print("   ÉVALUATION LSTM — Prédiction Météo Jena")
    print("=" * 60)

    # ── 1. Chargement modèle + données ───────────────────────────────
    print(f"\n[1/5] Chargement du modèle depuis '{args.model}' …")
    model = tf.keras.models.load_model(args.model)
    print("      Modèle chargé.")

    print("\n[2/5] Chargement des données …")
    df = load_jena_climate(data_dir=args.data)
    _, test_ds, scaler_target, target_idx, n_train, features, targets = \
        make_timeseries_datasets(df, batch_size=128)

    # ── 2. Prédictions ────────────────────────────────────────────────
    print("\n[3/5] Génération des prédictions …")
    t0 = time.time()
    y_pred_norm = model.predict(test_ds, verbose=1).flatten()
    infer_time  = round(time.time() - t0, 2)

    # Récupération des valeurs réelles (dé-normalisées)
    y_true_list = []
    for _, y_batch in test_ds:
        y_true_list.extend(y_batch.numpy().flatten())
    y_true_norm = np.array(y_true_list[:len(y_pred_norm)])

    # Dé-normalisation → vraies valeurs en °C
    y_pred_real = scaler_target.inverse_transform(
        y_pred_norm.reshape(-1, 1)).flatten()
    y_true_real = scaler_target.inverse_transform(
        y_true_norm.reshape(-1, 1)).flatten()

    print(f"      {len(y_pred_real)} prédictions en {infer_time}s")

    # ── 3. Métriques ──────────────────────────────────────────────────
    print("\n[4/5] Calcul des métriques …")
    mse  = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true_real, y_pred_real)
    r2   = r2_score(y_true_real, y_pred_real)
    mape = np.mean(np.abs((y_true_real - y_pred_real) /
                          (np.abs(y_true_real) + 1e-8))) * 100

    print(f"\n  MSE   : {mse:.6f}")
    print(f"  RMSE  : {rmse:.4f} °C")
    print(f"  MAE   : {mae:.4f} °C")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")

    # ── 4. Visualisations ─────────────────────────────────────────────
    print("\n[5/5] Génération des graphiques …")

    pred_path = os.path.join(args.output, "predictions_vs_reel.png")
    plot_predictions(y_true_real, y_pred_real, n_points=500,
                     save_path=pred_path)

    err_path = os.path.join(args.output, "distribution_erreurs.png")
    plot_error_distribution(y_true_real, y_pred_real,
                            save_path=err_path)

    # ── 5. Export JSON ────────────────────────────────────────────────
    report = {
        "model_path": args.model,
        "dataset":    "Jena Climate (météo horaire)",
        "target":     "T (degC) — Température T+1",
        "n_predictions": int(len(y_pred_real)),
        "inference_time_s": infer_time,
        "metrics": {
            "mse":  round(float(mse),  6),
            "rmse": round(float(rmse), 4),
            "mae":  round(float(mae),  4),
            "r2":   round(float(r2),   4),
            "mape_percent": round(float(mape), 2),
        },
        "sample_predictions": [
            {
                "index": i,
                "real_temp_C":  round(float(y_true_real[i]), 2),
                "pred_temp_C":  round(float(y_pred_real[i]), 2),
                "error_C":      round(float(y_pred_real[i] - y_true_real[i]), 3),
            }
            for i in range(min(20, len(y_pred_real)))
        ],
    }

    json_path = os.path.join(args.output, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  → JSON : {json_path}")

    # ── Résumé ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   RÉSUMÉ")
    print("=" * 60)
    print(f"  RMSE  : {rmse:.4f} °C  (erreur moyenne en degrés)")
    print(f"  MAE   : {mae:.4f} °C")
    print(f"  R²    : {r2:.4f}  (1.0 = parfait)")
    print(f"  MAPE  : {mape:.2f}%")
    print(f"\n  Fichiers dans '{args.output}/'")
    print("    - predictions_vs_reel.png")
    print("    - distribution_erreurs.png")
    print("    - evaluation_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
