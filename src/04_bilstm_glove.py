"""
Stage 4 — Rung 4: the enhancement rung. BiLSTM + GloVe + dropout + LR schedule.

Four changes over the plain LSTM of stage 3, each defensible in a viva:

  1. Bidirectional     — reads the sentence forwards and backwards, so the
                         representation of an early word can depend on a late one.
  2. GloVe init        — the embedding layer starts from vectors trained on 6B
                         tokens of Wikipedia instead of random noise. With ~10k
                         training sentences we cannot learn good embeddings from
                         scratch; this is transfer learning at the embedding layer.
  3. Dropout           — on the recurrent output and before the classifier.
  4. ReduceLROnPlateau — cuts the learning rate when dev loss stalls.

GloVe is fetched via gensim (~128 MB) and cached in artifacts/glove/, with the
Stanford zip as a fallback.

Run:  python src/04_bilstm_glove.py        (FORCE=1 to retrain)
"""
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

import src.common as U
import src.config as C

KEY = "bilstm_glove"
U.banner(f"STAGE 4 · RUNG 4 — {C.MODEL_BY_KEY[KEY]['name']}")


# ----------------------------------------------------------------------------
# GloVe acquisition
# ----------------------------------------------------------------------------
def glove_vectors() -> dict[str, np.ndarray]:
    """Return {word: vector}. Cached as a .npz after the first download."""
    cache = C.GLOVE_DIR / f"glove_{C.EMBED_DIM}d.npz"
    if cache.exists():
        print(f"  using cached {cache.name}")
        d = np.load(cache, allow_pickle=True)
        return dict(zip(d["words"].tolist(), d["vectors"]))

    vectors: dict[str, np.ndarray] = {}

    # Preferred route: gensim's downloader (smaller, resumable, no manual unzip)
    try:
        import gensim.downloader as api
        print(f"  downloading '{C.GLOVE_NAME}' via gensim "
              f"(~128 MB, first run only) …", flush=True)
        kv = api.load(C.GLOVE_NAME)
        vectors = {w: kv[w].astype("float32") for w in kv.index_to_key}
    except Exception as e:                                   # noqa: BLE001
        print(f"  gensim route unavailable ({type(e).__name__}: {e}).")
        print(f"  falling back to {C.GLOVE_FALLBACK_URL} (822 MB) …", flush=True)
        raw = C.GLOVE_DIR / "glove.6B.zip"
        if not raw.exists():
            with urlopen(C.GLOVE_FALLBACK_URL) as r, open(raw, "wb") as f:
                f.write(r.read())
        with zipfile.ZipFile(raw) as z:
            with z.open(f"glove.6B.{C.EMBED_DIM}d.txt") as fh:
                for line in io.TextIOWrapper(fh, encoding="utf-8"):
                    parts = line.rstrip().split(" ")
                    vectors[parts[0]] = np.asarray(parts[1:], dtype="float32")

    words = np.array(list(vectors.keys()), dtype=object)
    mat = np.stack([vectors[w] for w in words])
    np.savez_compressed(cache, words=words, vectors=mat)
    print(f"  cached {len(vectors):,} vectors → {cache.name}")
    return vectors


def build_embedding_matrix(vocab: dict[str, int]) -> tuple[np.ndarray, float]:
    gv = glove_vectors()
    rng = np.random.default_rng(C.SEED)
    # Unmatched words get small random vectors, PAD stays exactly zero.
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, C.EMBED_DIM)).astype("float32")
    matrix[U.PAD_ID] = 0.0
    hits = 0
    for word, idx in vocab.items():
        vec = gv.get(word)
        if vec is not None:
            matrix[idx] = vec
            hits += 1
    coverage = hits / max(len(vocab), 1)
    print(f"  GloVe covers {hits:,}/{len(vocab):,} vocabulary words ({coverage:.1%}); "
          f"the rest keep random init")
    return matrix, coverage


# ----------------------------------------------------------------------------
if not U.already_done(KEY, U.force_flag()):
    U.set_seeds()

    from tensorflow import keras
    from tensorflow.keras import layers
    import src.keras_common as K

    X, y, vocab, _ = K.prepare_arrays()
    n_classes = len(U.load_labels())
    emb_matrix, coverage = build_embedding_matrix(vocab)

    model = K.build_recurrent(
        layers.LSTM,
        vocab_size=len(vocab) + 2,
        n_classes=n_classes,
        units=C.BILSTM_UNITS,
        bidirectional=True,
        embedding_matrix=emb_matrix,
        trainable_embedding=C.GLOVE_TRAINABLE,
        recurrent_dropout_out=C.BILSTM_DROPOUT,
        dense_dropout=C.BILSTM_DENSE_DROPOUT,
    )

    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1,
    )

    K.train_and_record(
        KEY, model, X, y,
        extra_callbacks=[lr_schedule],
        extra_config={
            "units": C.BILSTM_UNITS,
            "bidirectional": True,
            "embeddings": f"GloVe {C.EMBED_DIM}d ({C.GLOVE_NAME})",
            "glove_coverage": round(coverage, 4),
            "embeddings_trainable": C.GLOVE_TRAINABLE,
            "dropout": C.BILSTM_DROPOUT,
            "dense_dropout": C.BILSTM_DENSE_DROPOUT,
            "lr_schedule": "ReduceLROnPlateau(factor=0.5, patience=2)",
            "batch_size": C.BATCH_SIZE,
            "max_len": C.MAX_LEN,
        },
    )

print("\n✅ Stage 4 done. Next: python src/05_distilbert.py")
