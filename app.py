"""
app.py — API Flask LSTM Météo Jena
====================================
Pages HTML :
  GET  /             → Interface prédiction (index.html)
  GET  /dashboard    → Dashboard modèle (dashboard.html)

Endpoints JSON :
  GET  /health       → État de l'API
  GET  /model/info   → Architecture & paramètres
  GET  /api/docs     → Documentation JSON
  POST /predict      → Prédiction T+1 (JSON : {temp, pres, rh, wv, wd})
  POST /predict/batch → Batch de prédictions (JSON : {observations: [...]})

Usage :
    python app.py
"""

import io
import time
import logging
import numpy as np

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from functools import wraps

# ── Import modèle custom (nécessaire pour la désérialisation Keras) ────────────
from models.rnn_model import WeatherLSTM  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_lstm_model.keras"

# Features d'entrée dans l'ordre exact utilisé à l'entraînement
FEATURE_NAMES = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "wd (deg)"]
FEATURE_KEYS  = ["temp", "pres", "rh", "wv", "wd"]          # clés JSON

# Bornes de validation (min, max) pour chaque feature
FEATURE_BOUNDS = {
    "temp": (-40.0,  60.0),
    "pres": (900.0, 1100.0),
    "rh":   (0.0,   100.0),
    "wv":   (0.0,    60.0),
    "wd":   (0.0,   360.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# Flask
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # 4 Mo

# ─────────────────────────────────────────────────────────────────────────────
# Modèle (singleton)
# ─────────────────────────────────────────────────────────────────────────────
model            = None
model_load_time  = None
model_load_error = None

# Scaler sauvegardé lors de l'entraînement (optionnel)
scaler_target    = None


def load_model():
    """Charge le modèle LSTM en mémoire et effectue un warm-up."""
    global model, model_load_time, model_load_error

    try:
        logger.info(f"Chargement du modèle depuis '{MODEL_PATH}' …")
        t0    = time.time()
        model = tf.keras.models.load_model(MODEL_PATH)

        # Warm-up : inférence fictive sur une séquence de 24 pas × 5 features
        dummy = np.zeros((1, 24, 5), dtype="float32")
        model.predict(dummy, verbose=0)

        model_load_time = round(time.time() - t0, 2)
        logger.info(f"Modèle prêt en {model_load_time}s")

    except Exception as e:
        model_load_error = str(e)
        logger.error(f"Erreur chargement modèle : {e}")


def _try_load_scaler():
    """Tente de charger le scaler scikit-learn (si disponible)."""
    global scaler_target
    import os, pickle
    paths = ["scaler_target.pkl", "utils/scaler_target.pkl"]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                scaler_target = pickle.load(f)
            logger.info(f"Scaler chargé depuis '{p}'")
            return


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if model is None:
            return jsonify({
                "error": "Modèle non disponible",
                "detail": model_load_error or "Pas encore chargé."
            }), 503
        return f(*args, **kwargs)
    return decorated


def validate_features(data: dict) -> tuple[list[float] | None, str | None]:
    """
    Valide et extrait les features depuis un dictionnaire JSON.
    Retourne (features_list, None) ou (None, message_erreur).
    """
    values = []
    for key in FEATURE_KEYS:
        if key not in data:
            return None, f"Champ requis manquant : '{key}'"
        try:
            v = float(data[key])
        except (TypeError, ValueError):
            return None, f"'{key}' doit être un nombre."
        lo, hi = FEATURE_BOUNDS[key]
        if not (lo <= v <= hi):
            return None, f"'{key}' hors limites [{lo}, {hi}]. Reçu : {v}"
        values.append(v)
    return values, None


def normalize_features(values: list[float]) -> np.ndarray:
    """
    Normalise manuellement les features dans [0,1] avec les bornes connues.
    Utilisé uniquement si le scaler n'est pas disponible.
    """
    lo = np.array([b[0] for b in FEATURE_BOUNDS.values()], dtype="float32")
    hi = np.array([b[1] for b in FEATURE_BOUNDS.values()], dtype="float32")
    arr = np.array(values, dtype="float32")
    return (arr - lo) / (hi - lo + 1e-8)


def build_sequence(values_norm: np.ndarray) -> np.ndarray:
    """
    Construit une séquence de 24 pas en répétant le vecteur courant.
    (Approximation pour la démo — en production, stocker un vrai historique.)
    """
    seq = np.tile(values_norm, (24, 1))     # (24, 5)
    return seq[np.newaxis, :, :]            # (1, 24, 5)


def denormalize_temperature(value_norm: float) -> float:
    """Dé-normalise la température prédite vers °C."""
    if scaler_target is not None:
        return float(scaler_target.inverse_transform([[value_norm]])[0][0])
    # Fallback : bornes manuelles
    lo, hi = FEATURE_BOUNDS["temp"]
    return float(value_norm * (hi - lo) + lo)


# ─────────────────────────────────────────────────────────────────────────────
# Routes HTML
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


# ─────────────────────────────────────────────────────────────────────────────
# Routes JSON
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    payload = {
        "status":       "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    }
    if model_load_time is not None:
        payload["model_load_time_s"] = model_load_time
    if model_load_error:
        payload["error"] = model_load_error
    return jsonify(payload), 200 if model else 503


@app.route("/model/info", methods=["GET"])
@model_required
def model_info():
    total = model.count_params()
    try:
        trainable = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    except Exception:
        trainable = None

    return jsonify({
        "architecture":          "WeatherLSTM",
        "task":                  "Régression — Prédiction température T+1",
        "dataset":               "Jena Climate 2009–2016",
        "sequence_length":       24,
        "n_features":            5,
        "feature_names":         FEATURE_NAMES,
        "target":                "T (degC)",
        "total_parameters":      total,
        "trainable_parameters":  trainable,
        "non_trainable_parameters": total - trainable if trainable else None,
        "layers": [
            {"name": l.name, "type": l.__class__.__name__}
            for l in model.layers
        ],
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Retourne les métriques d'évaluation depuis evaluation_results_lstm/evaluation_report.json.
    Généré par : python evaluate.py
    """
    import os, json
    paths = [
        "evaluation_results_lstm/evaluation_report.json",
        "evaluation_results/evaluation_report.json",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"available": True, "report": data}), 200

    return jsonify({
        "available": False,
        "message": "Aucun rapport trouvé. Lancez d'abord : python evaluate.py"
    }), 200


@app.route("/api/docs", methods=["GET"])
def api_docs():
    return jsonify({
        "name":    "LSTM Météo API",
        "version": "1.0.0",
        "dataset": "Jena Climate 2009–2016",
        "target":  "T (degC) — Température T+1",
        "features": dict(zip(FEATURE_KEYS, FEATURE_NAMES)),
        "endpoints": {
            "GET  /":              "Interface HTML — Prédiction",
            "GET  /dashboard":     "Interface HTML — Dashboard",
            "GET  /health":        "État de santé de l'API",
            "GET  /model/info":    "Architecture et paramètres du modèle",
            "GET  /api/docs":      "Cette documentation",
            "POST /predict":       (
                "Prédiction T+1. JSON : "
                "{'temp': 12.5, 'pres': 1013.0, 'rh': 75.0, 'wv': 3.5, 'wd': 180.0}"
            ),
            "POST /predict/batch": (
                "Batch de prédictions. JSON : "
                "{'observations': [{'temp':…,'pres':…,'rh':…,'wv':…,'wd':…}, …]}"
            ),
        },
        "feature_bounds": FEATURE_BOUNDS,
    }), 200


@app.route("/predict", methods=["POST"])
@model_required
def predict():
    t0 = time.time()

    if not request.is_json:
        return jsonify({"error": "Content-Type doit être application/json."}), 415

    data = request.get_json(silent=True) or {}
    values, err = validate_features(data)
    if err:
        return jsonify({"error": err}), 400

    # Normalisation + séquence
    values_norm = normalize_features(values)
    sequence    = build_sequence(values_norm)           # (1, 24, 5)

    # Inférence
    raw_pred  = float(model.predict(sequence, verbose=0)[0][0])
    temp_pred = denormalize_temperature(raw_pred)

    result = {
        "predicted_temperature_C": round(temp_pred, 4),
        "input":  dict(zip(FEATURE_KEYS, values)),
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
    }

    logger.info(
        f"/predict → T+1 = {temp_pred:.2f}°C "
        f"depuis T={values[0]:.1f}°C "
        f"({result['inference_time_ms']}ms)"
    )
    return jsonify(result), 200


@app.route("/predict/batch", methods=["POST"])
@model_required
def predict_batch():
    MAX_BATCH = 50
    t0 = time.time()

    if not request.is_json:
        return jsonify({"error": "application/json requis."}), 415

    data = request.get_json(silent=True) or {}
    observations = data.get("observations", [])

    if not isinstance(observations, list) or not observations:
        return jsonify({"error": "'observations' doit être une liste non vide."}), 400
    if len(observations) > MAX_BATCH:
        return jsonify({"error": f"Maximum {MAX_BATCH} observations. Reçu : {len(observations)}."}), 400

    sequences, errors, valid_inputs = [], [], []
    for i, obs in enumerate(observations):
        values, err = validate_features(obs)
        if err:
            errors.append({"index": i, "error": err})
            continue
        values_norm = normalize_features(values)
        sequences.append(build_sequence(values_norm)[0])  # (24, 5)
        valid_inputs.append(dict(zip(FEATURE_KEYS, values)))

    if not sequences:
        return jsonify({"error": "Aucune observation valide.", "details": errors}), 422

    batch    = np.stack(sequences)                           # (N, 24, 5)
    raw_preds = model.predict(batch, verbose=0).flatten()
    preds    = [round(denormalize_temperature(float(p)), 4) for p in raw_preds]

    return jsonify({
        "count":      len(preds),
        "predictions": [
            {
                "index": i,
                "input": valid_inputs[i],
                "predicted_temperature_C": preds[i],
            }
            for i in range(len(preds))
        ],
        "inference_time_ms": round((time.time() - t0) * 1000, 2),
        "errors": errors or None,
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# Erreurs globales
# ─────────────────────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route introuvable.", "detail": str(e)}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Méthode non autorisée."}), 405

@app.errorhandler(500)
def internal(e):
    logger.exception("Erreur interne")
    return jsonify({"error": "Erreur interne.", "detail": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _try_load_scaler()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)