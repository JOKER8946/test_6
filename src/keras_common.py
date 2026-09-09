"""Model builder + training loop shared by every Keras rung.

Kept in one place so SimpleRNN / GRU / LSTM / BiLSTM differ only by the things
the report claims they differ by.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF's startup noise

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import src.common as U
import src.config as C


def prepare_arrays():
    """Encode all three splits with the shared vocabulary."""
    d = U.load_all_splits()
    vocab = U.build_or_load_vocab()
    X = {s: U.encode_texts(d[s]["clean_text"], vocab) for s in U.SPLITS}
    y = {s: d[s]["label"].to_numpy() for s in U.SPLITS}
    return X, y, vocab, d


def build_recurrent(
    cell,
    vocab_size: int,
    n_classes: int,
    units: int = C.RNN_UNITS,
    bidirectional: bool = False,
    embedding_matrix: np.ndarray | None = None,
    trainable_embedding: bool = True,
    recurrent_dropout_out: float = 0.0,
    dense_dropout: float = 0.0,
    lr: float = C.LR,
) -> keras.Model:
    emb_kwargs = dict(
        input_dim=vocab_size,
        output_dim=C.EMBED_DIM,
        mask_zero=True,              # PAD (id 0) is ignored by the recurrent layer
        name="embedding",
    )
    if embedding_matrix is not None:
        emb_kwargs["embeddings_initializer"] = keras.initializers.Constant(embedding_matrix)
        emb_kwargs["trainable"] = trainable_embedding

    core = cell(units, dropout=recurrent_dropout_out, name="recurrent")
    if bidirectional:
        core = layers.Bidirectional(core, name="bidirectional")

    model = keras.Sequential([
        keras.Input(shape=(C.MAX_LEN,), dtype="int32"),
        layers.Embedding(**emb_kwargs),
        core,
        *( [layers.Dropout(dense_dropout, name="dropout")] if dense_dropout else [] ),
        layers.Dense(n_classes, activation="softmax", name="output"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_record(
    key: str,
    model: keras.Model,
    X, y,
    extra_callbacks: list | None = None,
    extra_config: dict | None = None,
) -> dict:
    """Fit with early stopping on dev, evaluate once on test, save everything."""
    meta = C.MODEL_BY_KEY[key]
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=C.PATIENCE,
            restore_best_weights=True, verbose=1,
        )
    ] + (extra_callbacks or [])

    with U.timed(f"training {meta['name']}") as t:
        hist = model.fit(
            X["train"], y["train"],
            validation_data=(X["dev"], y["dev"]),
            epochs=C.EPOCHS,
            batch_size=C.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

    dev_pred = model.predict(X["dev"], batch_size=256, verbose=0).argmax(1)
    test_pred = model.predict(X["test"], batch_size=256, verbose=0).argmax(1)
    dev_m = U.compute_metrics(y["dev"], dev_pred)
    test_m = U.compute_metrics(y["test"], test_pred)

    model_path = C.MODEL_DIR / f"{key}.keras"
    model.save(model_path)

    U.save_test_predictions(key, y["test"], test_pred)
    U.save_metrics(key, {
        "name": meta["name"],
        "family": meta["family"],
        "rung": meta["rung"],
        "dev": dev_m,
        "test": test_m,
        "train_seconds": t["seconds"],
        "model_size_mb": U.path_size_mb(model_path),
        "params": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()},
        "config": extra_config or {"units": C.RNN_UNITS, "batch_size": C.BATCH_SIZE,
                                   "lr": C.LR, "max_len": C.MAX_LEN},
    })

    print(f"\n  dev  macro-F1 {dev_m['macro_f1']:.4f} | accuracy {dev_m['accuracy']:.4f}")
    print(f"  test macro-F1 {test_m['macro_f1']:.4f} | accuracy {test_m['accuracy']:.4f}")
    print(f"  {model.count_params():,} parameters | best epoch "
          f"{np.argmin(hist.history['val_loss']) + 1} of {len(hist.history['loss'])}")
    return test_m
