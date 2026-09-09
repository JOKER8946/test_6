"""Shared helpers used by every stage of the pipeline."""
from __future__ import annotations

import json
import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import src.config as C

SPLITS = ("train", "dev", "test")

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seeds(seed: int = C.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Raw file discovery + column auto-detection
# ----------------------------------------------------------------------------
TEXT_CANDIDATES = ["text", "query", "sentence", "utterance", "question", "input"]
LABEL_CANDIDATES = ["category", "label", "intent", "class", "target", "y"]


def find_split_file(split: str) -> Path:
    """Locate data/<split>.xlsx (or .xls/.csv), case-insensitively."""
    for ext in (".xlsx", ".xls", ".csv"):
        for p in C.DATA_DIR.glob("*" + ext):
            if p.stem.lower() == split:
                return p
    # looser match: dev == validation/valid/val
    aliases = {"dev": ["validation", "valid", "val"], "test": ["eval"], "train": ["training"]}
    for alias in aliases.get(split, []):
        for ext in (".xlsx", ".xls", ".csv"):
            for p in C.DATA_DIR.glob("*" + ext):
                if p.stem.lower() == alias:
                    return p
    raise FileNotFoundError(
        f"Could not find a '{split}' file in {C.DATA_DIR}. "
        f"Expected {split}.xlsx (or .csv). Found: {[p.name for p in C.DATA_DIR.iterdir()] or 'nothing'}"
    )


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)          # needs openpyxl


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (text_column, label_column) whatever the Kaggle export called them."""
    lower = {c.lower().strip(): c for c in df.columns}

    text_col = next((lower[c] for c in TEXT_CANDIDATES if c in lower), None)
    label_col = next((lower[c] for c in LABEL_CANDIDATES if c in lower), None)

    if text_col is None or label_col is None:
        # Fall back on shape: the text column has the longest average string.
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if len(obj_cols) < 2:
            obj_cols = list(df.columns)
        scored = sorted(obj_cols, key=lambda c: df[c].astype(str).str.len().mean(), reverse=True)
        text_col = text_col or scored[0]
        label_col = label_col or next(c for c in scored[::-1] if c != text_col)

    return text_col, label_col


# ----------------------------------------------------------------------------
# Cleaning — deliberately minimal: no stopword removal, no lemmatisation.
# Banking77 intents hinge on function words ("card not working" vs "card
# working"), so the module's warning about stripping negations applies here.
# ----------------------------------------------------------------------------
_URL = re.compile(r"http\S+|www\.\S+")
_KEEP = re.compile(r"[^a-z0-9'\s]")
_WS = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = _URL.sub(" ", str(s).lower())
    s = _KEEP.sub(" ", s)
    return _WS.sub(" ", s).strip()


# ----------------------------------------------------------------------------
# Processed data access
# ----------------------------------------------------------------------------
def processed_path(split: str) -> Path:
    return C.PROC_DIR / f"{split}.csv"


def load_split(split: str) -> pd.DataFrame:
    p = processed_path(split)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `python src/01_data.py` first.")
    return pd.read_csv(p, keep_default_na=False)


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {s: load_split(s) for s in SPLITS}


def load_labels() -> list[str]:
    return json.loads((C.PROC_DIR / "labels.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Vocabulary shared by every Keras rung (built once, reused — keeps the
# rung-to-rung comparison honest)
# ----------------------------------------------------------------------------
PAD_ID, OOV_ID = 0, 1


def build_or_load_vocab(train_texts=None) -> dict[str, int]:
    path = C.PROC_DIR / "vocab.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if train_texts is None:
        raise FileNotFoundError("vocab.json missing — run `python src/01_data.py` first.")

    from collections import Counter
    counts = Counter(w for t in train_texts for w in str(t).split())
    keep = [w for w, _ in counts.most_common(C.MAX_VOCAB - 2)]
    vocab = {w: i + 2 for i, w in enumerate(keep)}      # 0 = PAD, 1 = OOV
    path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


def encode_texts(texts, vocab: dict[str, int], max_len: int = C.MAX_LEN) -> np.ndarray:
    out = np.zeros((len(texts), max_len), dtype="int32")
    for i, t in enumerate(texts):
        ids = [vocab.get(w, OOV_ID) for w in str(t).split()][:max_len]
        out[i, : len(ids)] = ids
    return out


# ----------------------------------------------------------------------------
# Metrics, artefacts, timing
# ----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "macro_f1": float(f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
    }


def metrics_path(key: str) -> Path:
    return C.METRIC_DIR / f"{key}.json"


def save_metrics(key: str, payload: dict) -> None:
    payload = {"key": key, **payload}
    metrics_path(key).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metrics(key: str) -> dict | None:
    p = metrics_path(key)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_test_predictions(key: str, y_true, y_pred) -> None:
    np.savez_compressed(
        C.PRED_DIR / f"{key}_test.npz",
        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
    )


def load_test_predictions(key: str):
    p = C.PRED_DIR / f"{key}_test.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_true"], d["y_pred"]


def path_size_mb(path: Path) -> float:
    path = Path(path)
    if path.is_file():
        return round(path.stat().st_size / 1e6, 2)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e6, 2)
    return 0.0


@contextmanager
def timed(label: str):
    """with timed('training') as t: ...   then t['seconds']"""
    box = {}
    t0 = time.perf_counter()
    print(f"  ⏱  {label} …", flush=True)
    yield box
    box["seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  ✔  {label} finished in {box['seconds']}s", flush=True)


def already_done(key: str, force: bool = False) -> bool:
    """Caching: skip a rung that has already produced metrics."""
    if force:
        return False
    if metrics_path(key).exists():
        m = load_metrics(key)
        print(f"⏭  '{key}' already trained "
              f"(test macro-F1 = {m['test']['macro_f1']:.4f}). "
              f"Delete {metrics_path(key).name} or set FORCE=1 to retrain.")
        return True
    return False


def force_flag() -> bool:
    return os.environ.get("FORCE", "").strip().lower() in {"1", "true", "yes"}


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)
