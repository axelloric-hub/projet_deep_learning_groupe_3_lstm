"""
utils/data_loader.py
====================
Chargement et préparation du dataset Météo de Jena (Keras).

Pipeline :
  1. Téléchargement automatique via tf.keras.utils.get_file
  2. Sélection des features météo
  3. Normalisation MinMaxScaler (Scikit-Learn)
  4. Construction des fenêtres temporelles (sliding window)
     via tf.keras.utils.timeseries_dataset_from_array
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import zipfile

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Features choisies parmi les colonnes Jena
FEATURE_COLUMNS = [
    "T (degC)",           # Température (cible)
    "p (mbar)",           # Pression atmosphérique
    "rh (%)",             # Humidité relative
    "wv (m/s)",           # Vitesse du vent
    "wd (deg)",           # Direction du vent
]
TARGET_COL = "T (degC)"   # Colonne à prédire

SEQUENCE_LENGTH = 24      # 24 pas de temps = 24 heures (données horaires)
TRAIN_RATIO     = 0.80    # 80% train, 20% test


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_jena_climate(data_dir: str = "data") -> pd.DataFrame:
    """
    Télécharge et charge le dataset Jena Climate.
    Retourne un DataFrame avec les features sélectionnées, sub-samplé à 1h.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")

    if not os.path.exists(csv_path):
        print("Téléchargement du dataset Jena Climate …")
        zip_path = tf.keras.utils.get_file(
            fname="jena_climate_2009_2016.csv.zip",
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            cache_dir=data_dir,
            cache_subdir="",
            extract=False,
        )
        # Extraction
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print(f"Dataset extrait → {csv_path}")

    df = pd.read_csv(csv_path)

    # Sub-sample : 1 ligne toutes les 6 (données originales = 10 min → 1 h)
    df = df[::6].reset_index(drop=True)

    # Garder uniquement les features utiles
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df[available].dropna()

    print(f"Dataset chargé : {len(df)} observations · {len(available)} features")
    print(f"Colonnes : {available}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame, train_size: int):
    """
    Normalise les données avec MinMaxScaler, ajusté sur le train uniquement.
    Retourne (data_scaled, scaler_target) où scaler_target permet
    d'inverser la normalisation sur la colonne cible.
    """
    scaler_all = MinMaxScaler()
    data_scaled = scaler_all.fit_transform(df[:train_size])

    # Scaler dédié à la colonne cible (pour inverse_transform sur les prédictions)
    target_idx = list(df.columns).index(TARGET_COL)
    scaler_target = MinMaxScaler()
    scaler_target.fit(df[TARGET_COL].values[:train_size].reshape(-1, 1))

    # Transformer le test avec le même scaler (pas de data leakage)
    test_scaled = scaler_all.transform(df[train_size:])
    all_scaled  = np.vstack([data_scaled, test_scaled])

    return all_scaled, scaler_target, target_idx


# ─────────────────────────────────────────────────────────────────────────────
# Fenêtres temporelles
# ─────────────────────────────────────────────────────────────────────────────

def make_timeseries_datasets(df: pd.DataFrame, batch_size: int = 64):
    """
    Construit les tf.data.Dataset train/test via sliding window.

    Retourne :
      train_ds, test_ds, scaler_target, target_idx, n_train
    Chaque batch : (batch, SEQUENCE_LENGTH, n_features) → (batch, 1)
    """
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)

    data_scaled, scaler_target, target_idx = normalize(df, n_train)

    features = data_scaled                       # (N, F)
    targets  = data_scaled[:, target_idx]        # (N,)

    # ── Train ─────────────────────────────────────────────────────────
    train_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=features[:n_train],
        targets=targets[SEQUENCE_LENGTH: n_train + 1],
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size,
    )

    # ── Test ──────────────────────────────────────────────────────────
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=features[n_train:],
        targets=targets[n_train + SEQUENCE_LENGTH: n_total + 1],
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=1,
        shuffle=False,
        batch_size=batch_size,
    )

    n_features = features.shape[1]
    print(f"Train : {n_train} obs | Test : {n_total - n_train} obs")
    print(f"Fenêtre : {SEQUENCE_LENGTH} pas | Features : {n_features}")

    return train_ds, test_ds, scaler_target, target_idx, n_train, features, targets
