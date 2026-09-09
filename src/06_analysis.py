"""
Stage 6 — Comparison table, figures and error analysis.

Reads whatever metrics exist (so you can run it after any rung), and produces:
  · artifacts/comparison.csv      — the table the examiner wants
  · artifacts/analysis.json       — confusion pairs + error examples for the app
  · artifacts/figures/*.png       — bar chart, training curves, confusion matrix

Run:  python src/06_analysis.py
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import src.common as U
import src.config as C

U.banner("STAGE 6 · COMPARISON, FIGURES AND ERROR ANALYSIS")

metrics = [m for m in (U.load_metrics(x["key"]) for x in C.MODELS) if m]
if not metrics:
    raise SystemExit("No metrics found — run stages 2-5 first.")
metrics.sort(key=lambda m: m["rung"])
print(f"Found {len(metrics)} of {len(C.MODELS)} rungs.")

labels = U.load_labels()
test_df = U.load_split("test")

# ----------------------------------------------------------------------------
# 1. Comparison table (quality + cost + deltas)
# ----------------------------------------------------------------------------
floor = metrics[0]["test"]["macro_f1"]
rows = []
prev = None
for m in metrics:
    t, d = m["test"], m["dev"]
    rows.append({
        "Rung": m["rung"],
        "Model": m["name"],
        "Family": m["family"],
        "Dev macro-F1": round(d["macro_f1"], 4),
        "Test macro-F1": round(t["macro_f1"], 4),
        "Test accuracy": round(t["accuracy"], 4),
        "Test macro-P": round(t["macro_precision"], 4),
        "Test macro-R": round(t["macro_recall"], 4),
        "Δ vs floor (pp)": round((t["macro_f1"] - floor) * 100, 2),
        "Δ vs previous (pp)": None if prev is None else round((t["macro_f1"] - prev) * 100, 2),
        "Params": m.get("params"),
        "Train time (s)": m.get("train_seconds"),
        "Size (MB)": m.get("model_size_mb"),
        "Epochs": m.get("epochs_run"),
    })
    prev = t["macro_f1"]

table = pd.DataFrame(rows)
table.to_csv(C.COMPARISON_CSV, index=False)
print("\n" + table[["Rung", "Model", "Test macro-F1", "Test accuracy",
                    "Δ vs previous (pp)", "Train time (s)", "Size (MB)"]]
      .to_string(index=False))

best = max(metrics, key=lambda m: m["test"]["macro_f1"])
print(f"\nBest rung: {best['name']} (test macro-F1 {best['test']['macro_f1']:.4f})")

# ----------------------------------------------------------------------------
# 2. Figures
# ----------------------------------------------------------------------------
INK, GRID, ACCENT, MUTED = "#1f2933", "#e3e8ee", "#2f6f8f", "#9aa5b1"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})

# 2a — macro-F1 across the ladder
fig, ax = plt.subplots(figsize=(8, 4))
names = [m["name"] for m in metrics]
vals = [m["test"]["macro_f1"] for m in metrics]
bars = ax.barh(names[::-1], vals[::-1],
               color=[ACCENT if v == max(vals) else MUTED for v in vals[::-1]])
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlim(0, min(1.0, max(vals) + 0.12))
ax.set_xlabel("Test macro-F1")
ax.set_title("Model ladder — test macro-F1", loc="left", fontweight="bold")
ax.grid(axis="y", visible=False)
fig.savefig(C.FIG_DIR / "macro_f1_ladder.png")
plt.close(fig)

# 2b — training curves
curved = [m for m in metrics if m.get("history")]
if curved:
    n = len(curved)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 5.4), squeeze=False)
    for j, m in enumerate(curved):
        h = m["history"]
        ep = range(1, len(h["loss"]) + 1)
        axes[0][j].plot(ep, h["loss"], color=ACCENT, label="train")
        axes[0][j].plot(ep, h["val_loss"], color="#c9743a", label="dev")
        axes[0][j].set_title(m["name"], fontsize=9)
        axes[0][j].set_xlabel("epoch")
        if j == 0:
            axes[0][j].set_ylabel("loss")
            axes[0][j].legend(frameon=False, fontsize=8)
        acc = h.get("accuracy") or []
        vacc = h.get("val_accuracy") or []
        if any(np.isfinite(acc)):
            axes[1][j].plot(ep, acc, color=ACCENT, label="train")
        if any(np.isfinite(vacc)):
            axes[1][j].plot(range(1, len(vacc) + 1), vacc, color="#c9743a", label="dev")
        axes[1][j].set_xlabel("epoch")
        if j == 0:
            axes[1][j].set_ylabel("accuracy")
    fig.suptitle("Training curves — overfitting shows as train/dev loss divergence",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "training_curves.png")
    plt.close(fig)

# 2c — confusion matrix for the best rung
pred = U.load_test_predictions(best["key"])
analysis = {"best_key": best["key"], "best_name": best["name"]}

if pred is not None:
    y_true, y_pred = pred
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(np.log1p(cm), cmap="Blues", interpolation="nearest")
    ax.set_title(f"Confusion matrix — {best['name']} (log scale, {len(labels)} classes)",
                 loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + count)")
    fig.savefig(C.FIG_DIR / "confusion_matrix.png")
    plt.close(fig)

    # Zoom: the 15 classes with the most errors, so the picture is readable
    err_per_class = cm.sum(1) - np.diag(cm)
    worst = np.argsort(err_per_class)[::-1][:15]
    sub = cm[np.ix_(worst, worst)]
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(len(worst)), [labels[i] for i in worst], rotation=90, fontsize=7)
    ax.set_yticks(range(len(worst)), [labels[i] for i in worst], fontsize=7)
    for a in range(len(worst)):
        for b in range(len(worst)):
            if sub[a, b]:
                ax.text(b, a, sub[a, b], ha="center", va="center", fontsize=6,
                        color="white" if sub[a, b] > sub.max() * 0.6 else INK)
    ax.set_title("15 hardest classes — where the errors actually are",
                 loc="left", fontweight="bold", fontsize=9)
    ax.grid(False)
    fig.savefig(C.FIG_DIR / "confusion_hard_classes.png")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Error analysis
    # ------------------------------------------------------------------
    wrong = np.flatnonzero(y_true != y_pred)
    pairs = Counter((int(y_true[i]), int(y_pred[i])) for i in wrong)

    confused = []
    for (t_id, p_id), count in pairs.most_common(12):
        idx = [int(i) for i in wrong if y_true[i] == t_id and y_pred[i] == p_id][:3]
        confused.append({
            "true": labels[t_id], "predicted": labels[p_id], "count": int(count),
            "examples": test_df.loc[idx, "text"].tolist(),
        })

    rep = classification_report(y_true, y_pred, labels=range(len(labels)),
                                target_names=labels, output_dict=True, zero_division=0)
    per_class = sorted(
        ({"label": l, "f1": rep[l]["f1-score"], "precision": rep[l]["precision"],
          "recall": rep[l]["recall"], "support": int(rep[l]["support"])} for l in labels),
        key=lambda r: r["f1"],
    )

    analysis.update({
        "n_test": int(len(y_true)),
        "n_wrong": int(len(wrong)),
        "error_rate": float(len(wrong) / len(y_true)),
        "confused_pairs": confused,
        "weakest_classes": per_class[:10],
        "strongest_classes": per_class[::-1][:10],
    })

    print(f"\n{len(wrong)} of {len(y_true)} test items misclassified "
          f"({len(wrong) / len(y_true):.2%})")
    print("Most-confused pairs:")
    for c in confused[:5]:
        print(f"  {c['count']:>3}×  {c['true']}  →  {c['predicted']}")
        print(f"        e.g. “{c['examples'][0]}”")

# Per-rung agreement with the best model: how much do the rungs actually differ?
agreement = {}
for m in metrics:
    p = U.load_test_predictions(m["key"])
    if p is not None and pred is not None:
        agreement[m["key"]] = float((p[1] == pred[1]).mean())
analysis["agreement_with_best"] = agreement

C.ANALYSIS_JSON.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
print(f"\nWrote {C.COMPARISON_CSV.name}, {C.ANALYSIS_JSON.name} and "
      f"{len(list(C.FIG_DIR.glob('*.png')))} figures.")
print("\n✅ Stage 6 done. Next: python src/07_report.py")
