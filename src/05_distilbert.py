"""
Stage 5 — Rung 5: fine-tuned DistilBERT.

The module's recipe exactly: distilbert-base-uncased, learning rate 2e-5,
4 epochs, weight decay 0.01, macro-F1 for model selection.

Why it is expected to win: the recurrent rungs learn their embeddings from
~10k sentences, and every embedding is static — one vector per word regardless
of context. DistilBERT arrives pre-trained on a large corpus, produces
contextual representations, and its WordPiece tokenizer has no out-of-vocabulary
failures (stage 1 printed the OOV rate the recurrent models are living with).

CPU note: this is the slow rung. Roughly 30-60 minutes per epoch on a typical
laptop CPU, so budget a few hours. Dynamic padding keeps it as cheap as it can
be. The run checkpoints each epoch into artifacts/models/distilbert_ckpt/, and
the finished model is cached, so this only happens once.

Run:  python src/05_distilbert.py         (FORCE=1 to retrain)
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

import src.common as U
import src.config as C

KEY = "distilbert"
U.banner(f"STAGE 5 · RUNG 5 — {C.MODEL_BY_KEY[KEY]['name']}")

if not U.already_done(KEY, U.force_flag()):
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    U.set_seeds()
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"  torch threads: {torch.get_num_threads()}  |  device: CPU")

    labels = U.load_labels()
    d = U.load_all_splits()

    tok = AutoTokenizer.from_pretrained(C.HF_MODEL)

    class IntentDataset(Dataset):
        """Tokenised once, padded per batch by the collator (much faster on CPU)."""

        def __init__(self, texts, targets):
            self.enc = tok(list(texts), truncation=True, max_length=C.BERT_MAX_LEN)
            self.targets = list(targets)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = int(self.targets[i])
            return item

    # NOTE: raw text, not clean_text — WordPiece was pre-trained on natural text,
    # and stripping punctuation would move the input away from what it expects.
    ds = {s: IntentDataset(d[s]["text"], d[s]["label"]) for s in U.SPLITS}

    model = AutoModelForSequenceClassification.from_pretrained(
        C.HF_MODEL,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )

    def metrics_fn(pred):
        return U.compute_metrics(pred.label_ids, pred.predictions.argmax(-1))

    ckpt_dir = C.MODEL_DIR / "distilbert_ckpt"
    base_args = dict(
        output_dir=str(ckpt_dir),
        num_train_epochs=C.BERT_EPOCHS,
        learning_rate=C.BERT_LR,             # small LR: don't wreck pre-trained weights
        per_device_train_batch_size=C.BERT_BATCH,
        per_device_eval_batch_size=C.BERT_EVAL_BATCH,
        weight_decay=C.BERT_WEIGHT_DECAY,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=C.SEED,
        use_cpu=True,
        report_to=[],
        disable_tqdm=False,
    )
    # transformers renamed evaluation_strategy → eval_strategy in 4.41. Accept either,
    # so this script survives a different transformers version than the pinned one.
    try:
        args = TrainingArguments(eval_strategy="epoch", **base_args)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **base_args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
    )

    with U.timed("fine-tuning DistilBERT (this is the long one)") as t:
        trainer.train()

    dev_out = trainer.predict(ds["dev"])
    test_out = trainer.predict(ds["test"])
    dev_m = U.compute_metrics(dev_out.label_ids, dev_out.predictions.argmax(-1))
    test_m = U.compute_metrics(test_out.label_ids, test_out.predictions.argmax(-1))

    save_dir = C.MODEL_DIR / KEY
    trainer.save_model(str(save_dir))
    tok.save_pretrained(str(save_dir))

    # Rebuild per-epoch curves from the Trainer log so the dashboard can plot
    # them next to the Keras histories.
    hist: dict[str, list[float]] = {"loss": [], "val_loss": [],
                                    "accuracy": [], "val_accuracy": []}
    train_losses: dict[int, list[float]] = {}
    for rec in trainer.state.log_history:
        ep = rec.get("epoch")
        if ep is None:
            continue
        if "loss" in rec:
            train_losses.setdefault(int(np.ceil(ep)), []).append(rec["loss"])
        if "eval_loss" in rec:
            hist["val_loss"].append(rec["eval_loss"])
            hist["val_accuracy"].append(rec.get("eval_accuracy", float("nan")))
            k = int(round(ep))
            batch = train_losses.get(k) or train_losses.get(k - 1) or [float("nan")]
            hist["loss"].append(float(np.mean(batch)))
    hist["accuracy"] = [float("nan")] * len(hist["loss"])   # HF logs no train accuracy

    U.save_test_predictions(KEY, test_out.label_ids, test_out.predictions.argmax(-1))
    U.save_metrics(KEY, {
        "name": C.MODEL_BY_KEY[KEY]["name"],
        "family": C.MODEL_BY_KEY[KEY]["family"],
        "rung": C.MODEL_BY_KEY[KEY]["rung"],
        "dev": dev_m,
        "test": test_m,
        "train_seconds": t["seconds"],
        "model_size_mb": U.path_size_mb(save_dir),
        "params": int(sum(p.numel() for p in model.parameters())),
        "epochs_run": C.BERT_EPOCHS,
        "history": hist,
        "config": {
            "checkpoint": C.HF_MODEL,
            "lr": C.BERT_LR,
            "epochs": C.BERT_EPOCHS,
            "batch_size": C.BERT_BATCH,
            "max_length": C.BERT_MAX_LEN,
            "weight_decay": C.BERT_WEIGHT_DECAY,
            "tokenizer": "WordPiece (subword — no OOV)",
        },
    })

    print(f"\n  dev  macro-F1 {dev_m['macro_f1']:.4f} | accuracy {dev_m['accuracy']:.4f}")
    print(f"  test macro-F1 {test_m['macro_f1']:.4f} | accuracy {test_m['accuracy']:.4f}")

    # The epoch checkpoints are large and no longer needed once the best model
    # is saved. Comment this out if you want to inspect them.
    import shutil
    shutil.rmtree(ckpt_dir, ignore_errors=True)

print("\n✅ Stage 5 done. Next: python src/06_analysis.py")
