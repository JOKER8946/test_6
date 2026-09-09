"""
Stage 2 — Rung 0: TF-IDF + Logistic Regression.

The classical floor. Sparse word/bigram counts weighted by inverse document
frequency, no word order, no semantics: "loan not approved" and "not loan
approved" are the same vector. Every neural rung has to beat this number to
justify its cost.

Run:  python src/02_tfidf.py        (FORCE=1 to retrain)
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.common as U
import src.config as C

KEY = "tfidf_lr"
U.banner(f"STAGE 2 · RUNG 0 — {C.MODEL_BY_KEY[KEY]['name']}")

if not U.already_done(KEY, U.force_flag()):
    U.set_seeds()
    d = U.load_all_splits()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # bigrams recover a sliver of word order
            min_df=1,
            sublinear_tf=True,       # log(1+tf) damps repeated words
            max_features=50_000,
        )),
        ("clf", LogisticRegression(
            C=5.0, max_iter=2000, random_state=C.SEED,
        )),
    ])

    with U.timed("fitting TF-IDF + LogisticRegression") as t:
        pipe.fit(d["train"]["clean_text"], d["train"]["label"])

    dev_pred = pipe.predict(d["dev"]["clean_text"])
    test_pred = pipe.predict(d["test"]["clean_text"])

    dev_m = U.compute_metrics(d["dev"]["label"], dev_pred)
    test_m = U.compute_metrics(d["test"]["label"], test_pred)

    model_path = C.MODEL_DIR / f"{KEY}.joblib"
    joblib.dump(pipe, model_path)

    n_features = len(pipe.named_steps["tfidf"].vocabulary_)
    n_params = n_features * len(U.load_labels())

    U.save_test_predictions(KEY, d["test"]["label"], test_pred)
    U.save_metrics(KEY, {
        "name": C.MODEL_BY_KEY[KEY]["name"],
        "family": C.MODEL_BY_KEY[KEY]["family"],
        "rung": C.MODEL_BY_KEY[KEY]["rung"],
        "dev": dev_m,
        "test": test_m,
        "train_seconds": t["seconds"],
        "model_size_mb": U.path_size_mb(model_path),
        "params": int(n_params),
        "epochs_run": None,
        "history": None,
        "config": {"ngram_range": "1-2", "max_features": 50000, "C": 5.0},
    })

    print(f"\n  dev  macro-F1 {dev_m['macro_f1']:.4f} | accuracy {dev_m['accuracy']:.4f}")
    print(f"  test macro-F1 {test_m['macro_f1']:.4f} | accuracy {test_m['accuracy']:.4f}")
    print(f"  {n_features:,} TF-IDF features → {n_params:,} weights, "
          f"{U.path_size_mb(model_path)} MB on disk")

print("\n✅ Stage 2 done. Next: python src/03_recurrent.py")
