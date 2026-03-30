"""
utils/visualisation.py
=======================
Fonctions de visualisation pour le modèle LSTM météo.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# ─────────────────────────────────────────────────────────────────────────────
# Style commun
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#2a2a3a",
    "axes.labelcolor":   "#c0c0d8",
    "xtick.color":       "#7070a0",
    "ytick.color":       "#7070a0",
    "text.color":        "#e8e8f0",
    "grid.color":        "#2a2a3a",
    "grid.linestyle":    ":",
    "grid.alpha":        0.6,
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Courbes d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history, save_path: str = "courbes_lstm.png"):
    """
    Trace Train Loss vs Validation Loss (MSE) extrait de l'objet History.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'entraînement — LSTM Météo Jena",
                 fontsize=13, fontweight="bold", color="#e8e8f0")

    epochs = range(1, len(history.history["loss"]) + 1)
    color_train = "#6ee7f7"
    color_val   = "#f7c06e"

    # Loss (MSE)
    ax1 = axes[0]
    ax1.plot(epochs, history.history["loss"],     color=color_train, lw=2, label="Train MSE")
    ax1.plot(epochs, history.history["val_loss"], color=color_val,   lw=2, ls="--", label="Val MSE")
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("MSE")
    ax1.set_title("Train Loss vs Validation Loss", color="#e8e8f0")
    ax1.legend(facecolor="#1a1a2a", edgecolor="#2a2a3a")
    ax1.grid(True)

    # MAE (si disponible)
    ax2 = axes[1]
    if "mae" in history.history:
        ax2.plot(epochs, history.history["mae"],     color=color_train, lw=2, label="Train MAE")
        ax2.plot(epochs, history.history["val_mae"], color=color_val,   lw=2, ls="--", label="Val MAE")
        ax2.set_ylabel("MAE")
        ax2.set_title("Train MAE vs Validation MAE", color="#e8e8f0")
    else:
        ax2.plot(epochs, history.history["loss"],     color=color_train, lw=2, label="Train MSE")
        ax2.plot(epochs, history.history["val_loss"], color=color_val,   lw=2, ls="--", label="Val MSE")
        ax2.set_ylabel("MSE")
        ax2.set_title("Convergence", color="#e8e8f0")
    ax2.set_xlabel("Époque")
    ax2.legend(facecolor="#1a1a2a", edgecolor="#2a2a3a")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Courbes sauvegardées : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prédictions vs Réalité (requis par le sujet)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     n_points: int = 500,
                     save_path: str = "predictions_vs_reel.png"):
    """
    Superpose la courbe réelle et la courbe prédite sur les n_points premiers
    points du jeu de test (valeurs dé-normalisées en °C).
    """
    _apply_style()
    y_true = y_true[:n_points]
    y_pred = y_pred[:n_points]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Prédictions LSTM vs Températures Réelles — Jena ({n_points} heures)",
        fontsize=13, fontweight="bold", color="#e8e8f0"
    )

    # ── Courbe principale ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(y_true, color="#6ee7f7", lw=1.5, alpha=0.9, label="Température réelle (°C)")
    ax1.plot(y_pred, color="#f7c06e", lw=1.5, alpha=0.9, ls="--", label="Prédiction LSTM (°C)")
    ax1.fill_between(range(len(y_true)), y_true, y_pred,
                     color="#f76e6e", alpha=0.08, label="Écart")
    ax1.set_ylabel("Température (°C)")
    ax1.legend(facecolor="#1a1a2a", edgecolor="#2a2a3a", fontsize=10)
    ax1.grid(True)

    # ── Résidus ───────────────────────────────────────────────────────
    ax2 = axes[1]
    residus = y_pred - y_true
    ax2.bar(range(len(residus)), residus,
            color=np.where(residus >= 0, "#6ef7a0", "#f76e6e"),
            alpha=0.7, width=1.0)
    ax2.axhline(0, color="#7070a0", lw=1)
    ax2.set_xlabel("Pas de temps (heures)")
    ax2.set_ylabel("Résidu (°C)")
    ax2.set_title("Erreurs de prédiction (résidus)", color="#e8e8f0", fontsize=10)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Graphique prédictions sauvegardé : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Distribution des erreurs
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: str = "distribution_erreurs.png"):
    """Distribution des résidus et scatter réel vs prédit."""
    _apply_style()
    residus = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Analyse des erreurs — LSTM Météo",
                 fontsize=13, fontweight="bold", color="#e8e8f0")

    # Histogramme des résidus
    ax1 = axes[0]
    ax1.hist(residus, bins=60, color="#6ee7f7", edgecolor="#0f1117", alpha=0.85)
    ax1.axvline(0, color="#f7c06e", lw=2, ls="--", label="Résidu = 0")
    ax1.axvline(residus.mean(), color="#f76e6e", lw=1.5, ls=":",
                label=f"Moyenne = {residus.mean():.3f}")
    ax1.set_xlabel("Résidu (°C)")
    ax1.set_ylabel("Fréquence")
    ax1.set_title("Distribution des résidus", color="#e8e8f0")
    ax1.legend(facecolor="#1a1a2a", edgecolor="#2a2a3a", fontsize=9)
    ax1.grid(True)

    # Scatter réel vs prédit
    ax2 = axes[1]
    sample = min(2000, len(y_true))
    idx = np.random.choice(len(y_true), sample, replace=False)
    ax2.scatter(y_true[idx], y_pred[idx],
                alpha=0.35, s=8, color="#6ee7f7")
    lim = [min(y_true.min(), y_pred.min()) - 1,
           max(y_true.max(), y_pred.max()) + 1]
    ax2.plot(lim, lim, color="#f7c06e", lw=1.5, ls="--", label="Parfait (y=x)")
    ax2.set_xlabel("Température réelle (°C)")
    ax2.set_ylabel("Température prédite (°C)")
    ax2.set_title("Réel vs Prédit", color="#e8e8f0")
    ax2.legend(facecolor="#1a1a2a", edgecolor="#2a2a3a", fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.show()
    print(f"Distribution des erreurs sauvegardée : {save_path}")
