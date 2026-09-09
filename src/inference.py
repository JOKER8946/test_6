"""
One prediction interface over six very different models.

Each rung stores itself differently — a scikit-learn pipeline, three Keras
models sharing a word vocabulary, and a HuggingFace directory — so this module
hides that behind `predict(key, text)` and the dashboard stays simple.
"""
from __future__ import annotations

import time

import numpy as np

import src.common as U
import src.config as C

_CACHE: dict[str, object] = {}

KERAS_KEYS = {"simple_rnn", "gru", "lstm", "bilstm_glove"}


def available_models() -> list[dict]:
    """The rungs that have both a trained artefact and recorded metrics."""
    out = []
    for m in C.MODELS:
        k = m["key"]
        exists = (
            (C.MODEL_DIR / f"{k}.joblib").exists()
            or (C.MODEL_DIR / f"{k}.keras").exists()
            or (C.MODEL_DIR / k).is_dir()
        )
        if exists and U.metrics_path(k).exists():
            out.append(m)
    return out


def _load(key: str):
    if key in _CACHE:
        return _CACHE[key]

    if key == "tfidf_lr":
        import joblib
        obj = joblib.load(C.MODEL_DIR / "tfidf_lr.joblib")

    elif key in KERAS_KEYS:
        from tensorflow import keras
        obj = (keras.models.load_model(C.MODEL_DIR / f"{key}.keras"),
               U.build_or_load_vocab())

    elif key == "distilbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = str(C.MODEL_DIR / "distilbert")
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        torch.set_num_threads(2)
        obj = (model, AutoTokenizer.from_pretrained(path))

    else:
        raise KeyError(f"Unknown model key: {key}")

    _CACHE[key] = obj
    return obj


def predict(key: str, text: str, top_k: int = 5) -> dict:
    """Return {'top': [(label, prob), ...], 'latency_ms': float}."""
    labels = U.load_labels()
    t0 = time.perf_counter()

    if key == "tfidf_lr":
        pipe = _load(key)
        probs = pipe.predict_proba([U.clean_text(text)])[0]

    elif key in KERAS_KEYS:
        model, vocab = _load(key)
        x = U.encode_texts([U.clean_text(text)], vocab)
        probs = model.predict(x, verbose=0)[0]

    elif key == "distilbert":
        import torch
        model, tok = _load(key)
        enc = tok(text, truncation=True, max_length=C.BERT_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1).numpy()

    else:
        raise KeyError(key)

    latency = (time.perf_counter() - t0) * 1000
    order = np.argsort(probs)[::-1][:top_k]
    return {
        "top": [(labels[i], float(probs[i])) for i in order],
        "latency_ms": round(latency, 1),
    }


def predict_all(text: str, keys: list[str] | None = None, top_k: int = 3) -> dict[str, dict]:
    keys = keys or [m["key"] for m in available_models()]
    return {k: predict(k, text, top_k=top_k) for k in keys}
