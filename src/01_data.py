"""
Stage 1 — Data.

Reads data/train.xlsx, data/dev.xlsx, data/test.xlsx (column names auto-detected),
cleans the text, encodes the 77 intent labels, and writes everything the later
stages need into artifacts/processed/.

Protocol: train = fitting, dev = validation / early stopping / model selection,
test = touched once per rung for the final number. Nothing is tuned on test.

Run:  python src/01_data.py
"""
import json

import numpy as np
import pandas as pd

import src.common as U
import src.config as C

U.banner("STAGE 1 · DATA PREPARATION")
U.set_seeds()

# ----------------------------------------------------------------------------
# 1. Read the three Excel files
# ----------------------------------------------------------------------------
raw = {}
for split in U.SPLITS:
    path = U.find_split_file(split)
    df = U._read_any(path)
    text_col, label_col = U.detect_columns(df)
    print(f"{split:>5}: {path.name:<20} rows={len(df):<6} "
          f"text column='{text_col}'  label column='{label_col}'")
    raw[split] = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label_name": df[label_col].astype(str).str.strip(),
    })

# ----------------------------------------------------------------------------
# 2. Clean + drop empties/duplicates (duplicates only inside train)
# ----------------------------------------------------------------------------
for split, df in raw.items():
    df["clean_text"] = df["text"].map(U.clean_text)
    before = len(df)
    df.drop(df.index[df["clean_text"].str.len() == 0], inplace=True)
    if split == "train":
        df.drop_duplicates(subset=["clean_text", "label_name"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if before != len(df):
        print(f"  {split}: {before - len(df)} empty/duplicate rows removed")

# ----------------------------------------------------------------------------
# 3. Label encoding — the label space is defined by train
# ----------------------------------------------------------------------------
labels = sorted(raw["train"]["label_name"].unique())
label2id = {l: i for i, l in enumerate(labels)}
print(f"\nClasses: {len(labels)}")

for split, df in raw.items():
    unseen = set(df["label_name"]) - set(label2id)
    if unseen:
        print(f"  ⚠ {split} contains {len(unseen)} labels absent from train; "
              f"those rows are dropped: {sorted(unseen)[:5]}")
        df.drop(df.index[df["label_name"].isin(unseen)], inplace=True)
        df.reset_index(drop=True, inplace=True)
    df["label"] = df["label_name"].map(label2id).astype(int)

(C.PROC_DIR / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------------
# 4. Shared vocabulary for the Keras rungs (built from train only)
# ----------------------------------------------------------------------------
(C.PROC_DIR / "vocab.json").unlink(missing_ok=True)
vocab = U.build_or_load_vocab(raw["train"]["clean_text"].tolist())
print(f"Vocabulary: {len(vocab):,} words kept (cap {C.MAX_VOCAB:,}), "
      f"+ PAD and OOV slots")

# OOV rate on dev — an honest look at what the static vocabulary misses
dev_words = [w for t in raw["dev"]["clean_text"] for w in t.split()]
oov_rate = sum(w not in vocab for w in dev_words) / max(len(dev_words), 1)
print(f"Out-of-vocabulary rate on dev: {oov_rate:.2%} "
      f"(DistilBERT's subword tokenizer drives this to 0)")

# ----------------------------------------------------------------------------
# 5. Length statistics — this is what MAX_LEN should be based on
# ----------------------------------------------------------------------------
lens = raw["train"]["clean_text"].str.split().str.len()
pct = {f"p{q}": int(np.percentile(lens, q)) for q in (50, 90, 95, 99)}
print(f"\nToken counts (train): mean={lens.mean():.1f}  max={lens.max()}  " +
      "  ".join(f"{k}={v}" for k, v in pct.items()))
print(f"MAX_LEN in config.py = {C.MAX_LEN} "
      f"→ covers {(lens <= C.MAX_LEN).mean():.2%} of training sentences without truncation")

counts = raw["train"]["label_name"].value_counts()
print(f"Class balance (train): smallest={counts.min()}  largest={counts.max()}  "
      f"ratio={counts.max() / counts.min():.2f}×  → macro-F1 is the primary metric")

# ----------------------------------------------------------------------------
# 6. Save
# ----------------------------------------------------------------------------
for split, df in raw.items():
    df[["text", "clean_text", "label", "label_name"]].to_csv(
        U.processed_path(split), index=False
    )
    print(f"  → {U.processed_path(split).relative_to(C.ROOT)}  ({len(df):,} rows)")

stats = {
    "n_classes": len(labels),
    "rows": {s: int(len(df)) for s, df in raw.items()},
    "vocab_size": len(vocab) + 2,
    "oov_rate_dev": float(oov_rate),
    "token_len": {"mean": float(lens.mean()), "max": int(lens.max()), **pct},
    "max_len_coverage": float((lens <= C.MAX_LEN).mean()),
    "class_counts": {str(k): int(v) for k, v in counts.items()},
    "imbalance_ratio": float(counts.max() / counts.min()),
    "examples": raw["train"].sample(8, random_state=C.SEED)[["text", "label_name"]]
                 .to_dict("records"),
}
C.DATASET_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

print("\n✅ Stage 1 done. Next: python src/02_tfidf.py")
