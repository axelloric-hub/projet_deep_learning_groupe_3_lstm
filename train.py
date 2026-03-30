"""
train.py — Entraînement LSTM Météo Jena
========================================
Usage :
    python train.py
    python train.py --epochs 80 --batch 128 --lr 0.001
"""
 
import argparse
import os
import numpy as np
import tensorflow as tf
 
from models.rnn_model import WeatherLSTM
from utils.data_loader import load_jena_climate, make_timeseries_datasets, SEQUENCE_LENGTH
from utils.visualisation import plot_history
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Hyperparamètres par défaut
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "epochs":      100,
    "batch_size":  64,
    "lr":          1e-3,
    "lstm1":       128,
    "lstm2":       64,
    "dense":       32,
    "dropout":     0.2,
    "patience_es": 10,
    "patience_lr": 5,
    "model_path":  "best_lstm_model.keras",
    "data_dir":    "data",
}
 
 
def parse_args():
    p = argparse.ArgumentParser(description="Entraînement LSTM — Météo Jena")
    p.add_argument("--epochs",    type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",        type=float, default=DEFAULTS["lr"])
    p.add_argument("--lstm1",     type=int,   default=DEFAULTS["lstm1"])
    p.add_argument("--lstm2",     type=int,   default=DEFAULTS["lstm2"])
    p.add_argument("--dense",     type=int,   default=DEFAULTS["dense"])
    p.add_argument("--dropout",     type=float, default=DEFAULTS["dropout"])
    p.add_argument("--patience_es", type=int,   default=DEFAULTS["patience_es"])
    p.add_argument("--patience_lr", type=int,   default=DEFAULTS["patience_lr"])
    p.add_argument("--model",       default=DEFAULTS["model_path"])
    p.add_argument("--data_dir",    default=DEFAULTS["data_dir"])
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    print("=" * 60)
    print("   Entraînement LSTM — Prédiction Météo (Jena Climate)")
    print("=" * 60)
 
    # ── 1. Données ────────────────────────────────────────────────────
    df = load_jena_climate(data_dir=args.data_dir)
    train_ds, test_ds, scaler_target, target_idx, n_train, features, targets = \
        make_timeseries_datasets(df, batch_size=args.batch)
 
    n_features = features.shape[1]
    print(f"\nFeatures d'entrée : {n_features}")
    print(f"Longueur de séquence : {SEQUENCE_LENGTH}")
 
    # ── 2. Modèle ─────────────────────────────────────────────────────
    model = WeatherLSTM(
        lstm_units_1=args.lstm1,
        lstm_units_2=args.lstm2,
        dense_units=args.dense,
        dropout_rate=args.dropout,
    )
 
    # Build pour afficher le summary
    model.build(input_shape=(None, SEQUENCE_LENGTH, n_features))
    model.summary()
 
    # ── 3. Compilation ────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
 
    # ── 4. Callbacks ──────────────────────────────────────────────────
    callbacks = [
        # Arrêt anticipé si val_loss stagne
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        # Réduction LR si stagnation
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=args.patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
        # Sauvegarde du meilleur modèle
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        # Logs lisibles dans le terminal
        tf.keras.callbacks.TerminateOnNaN(),
    ]
 
    # ── 5. Entraînement ───────────────────────────────────────────────
    print(f"\nDébut de l'entraînement (max {args.epochs} époques) …\n")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=test_ds,
        callbacks=callbacks,
    )
 
    # ── 6. Évaluation rapide ──────────────────────────────────────────
    print("\n--- Évaluation sur le jeu de test ---")
    results = model.evaluate(test_ds, verbose=1)
    mse = results[0]
    mae = results[1] if len(results) > 1 else None
    print(f"\nTest MSE : {mse:.6f}")
    print(f"Test RMSE: {mse**0.5:.4f} °C")
    if mae:
        print(f"Test MAE : {mae:.4f} °C")
 
    # ── 7. Courbes d'entraînement ─────────────────────────────────────
    plot_history(history, save_path="courbes_lstm.png")
 
    print(f"\nModèle sauvegardé : '{args.model}'")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 