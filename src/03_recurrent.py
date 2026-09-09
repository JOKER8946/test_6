"""
Stage 3 — Rungs 1-3: SimpleRNN → GRU → LSTM.

Identical everything (vocabulary, embedding size, units, optimiser, batch size,
early stopping) except the recurrent cell. That is the whole point: any
difference in the table is attributable to gating, not to tuning.

Embeddings are randomly initialised and learned end-to-end here — the GloVe
comparison arrives in stage 4.

Each cell is cached separately, so a crash or a Ctrl-C on LSTM does not cost
you the GRU run.

Run:  python src/03_recurrent.py          (FORCE=1 to retrain all three)
"""
from tensorflow.keras import layers

import src.common as U
import src.config as C
import src.keras_common as K

CELLS = [
    ("simple_rnn", layers.SimpleRNN),
    ("gru", layers.GRU),
    ("lstm", layers.LSTM),
]

U.banner("STAGE 3 · RUNGS 1-3 — SimpleRNN vs GRU vs LSTM")

force = U.force_flag()
pending = [(k, c) for k, c in CELLS if not U.already_done(k, force)]

if pending:
    U.set_seeds()
    X, y, vocab, _ = K.prepare_arrays()
    n_classes = len(U.load_labels())
    vocab_size = len(vocab) + 2

    for key, cell in pending:
        U.banner(f"RUNG {C.MODEL_BY_KEY[key]['rung']} — {C.MODEL_BY_KEY[key]['name']}")
        U.set_seeds()                      # same init conditions for every cell
        model = K.build_recurrent(cell, vocab_size, n_classes)
        K.train_and_record(key, model, X, y)

print("\n✅ Stage 3 done. Next: python src/04_bilstm_glove.py")
print("   Expected shape of the result: SimpleRNN clearly lowest, GRU ≈ LSTM,")
print("   GRU trains faster (fewer gates, fewer parameters).")
