"""
Test-6 dashboard.

    streamlit run app.py

Reads only what the pipeline wrote into artifacts/, so whatever has finished
training is what shows up. Nothing here trains anything.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import src.common as U
import src.config as C

st.set_page_config(page_title="Test-6 · Banking77 model ladder",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# Look: light, quiet, one accent. Charts carry the page, not the chrome.
# ----------------------------------------------------------------------------
ACCENT, INK, SUBTLE, RULE = "#2f6f8f", "#1f2933", "#616e7c", "#e1e7ec"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, [class*="css"] {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:{INK}; }}
h1, h2, h3 {{ font-family:'Source Serif 4', Georgia, serif; font-weight:600; letter-spacing:-0.01em; }}
h1 {{ font-size:2.1rem; margin-bottom:0.1rem; }}
.block-container {{ padding-top:2.4rem; max-width:1180px; }}
.lede {{ color:{SUBTLE}; font-size:1.02rem; max-width:68ch; line-height:1.55; }}
.stat {{ border-left:3px solid {ACCENT}; padding:0.15rem 0 0.15rem 0.85rem; }}
.stat .v {{ font-family:'Source Serif 4', Georgia, serif; font-size:1.85rem; line-height:1.15; }}
.stat .k {{ color:{SUBTLE}; font-size:0.82rem; }}
.note {{ color:{SUBTLE}; font-size:0.88rem; line-height:1.5; max-width:72ch; }}
hr {{ border:none; border-top:1px solid {RULE}; margin:1.6rem 0; }}
[data-testid="stMetricValue"] {{ font-family:'Source Serif 4', Georgia, serif; }}
</style>
""", unsafe_allow_html=True)


def stat(value, key):
    st.markdown(f'<div class="stat"><div class="v">{value}</div>'
                f'<div class="k">{key}</div></div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_everything():
    metrics = [m for m in (U.load_metrics(x["key"]) for x in C.MODELS) if m]
    metrics.sort(key=lambda m: m["rung"])
    stats = json.loads(C.DATASET_STATS.read_text()) if C.DATASET_STATS.exists() else None
    analysis = json.loads(C.ANALYSIS_JSON.read_text()) if C.ANALYSIS_JSON.exists() else None
    table = pd.read_csv(C.COMPARISON_CSV) if C.COMPARISON_CSV.exists() else None
    labels = U.load_labels() if (C.PROC_DIR / "labels.json").exists() else []
    return metrics, stats, analysis, table, labels


@st.cache_data(show_spinner=False)
def load_test_frame():
    try:
        return U.load_split("test")
    except FileNotFoundError:
        return None


metrics, stats, analysis, table, labels = load_everything()

if not metrics:
    st.title("Test-6 · Banking77 model ladder")
    st.warning("No results yet. Run the pipeline first:\n\n"
               "```\npython src/01_data.py\npython src/02_tfidf.py\n"
               "python src/03_recurrent.py\npython src/04_bilstm_glove.py\n"
               "python src/05_distilbert.py\npython src/06_analysis.py\n```")
    st.stop()

by_key = {m["key"]: m for m in metrics}
best = max(metrics, key=lambda m: m["test"]["macro_f1"])
floor = metrics[0]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Run status")
    for m in C.MODELS:
        done = m["key"] in by_key
        score = f"{by_key[m['key']]['test']['macro_f1']:.3f}" if done else "—"
        st.markdown(f"{'●' if done else '○'} {m['name']} &nbsp;`{score}`",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Primary metric: macro-F1 · {stats['n_classes'] if stats else '?'} classes · "
               f"seed {C.SEED} · CPU")
    if C.REPORT_PATH.exists():
        st.download_button("Download REPORT.md", C.REPORT_PATH.read_bytes(),
                           file_name="REPORT.md", use_container_width=True)
    if C.COMPARISON_CSV.exists():
        st.download_button("Download comparison.csv", C.COMPARISON_CSV.read_bytes(),
                           file_name="comparison.csv", use_container_width=True)

# ----------------------------------------------------------------------------
st.title("From bag-of-words to Transformer, one change at a time")
note("Six models on the same Banking77 split, the same seed and the same metric. "
     "Each rung changes one idea from Module 5 — memory, gating, pre-trained "
     "embeddings, contextual embeddings — so the gap between two rows is "
     "attributable to that idea rather than to tuning.")
st.write("")

tabs = st.tabs(["Results", "Dataset", "Training", "Errors", "Try it"])

# ============================================================== RESULTS ======
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat(f"{best['test']['macro_f1']:.3f}", f"best macro-F1 · {best['name']}")
    with c2:
        stat(f"{(best['test']['macro_f1'] - floor['test']['macro_f1']) * 100:+.1f} pp",
             "gain over the classical floor")
    with c3:
        stat(f"{best['test']['accuracy']:.3f}", "test accuracy, best model")
    with c4:
        secs = sum(m.get("train_seconds") or 0 for m in metrics)
        stat(f"{secs/60:.0f} min", "total CPU training time")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Altair field names stay simple (no hyphens or spaces) — shorthand parsing
    # is fussy about both. The display table renames them afterwards.
    df = pd.DataFrame([{
        "model": m["name"], "rung": m["rung"], "family": m["family"],
        "macro_f1": m["test"]["macro_f1"], "accuracy": m["test"]["accuracy"],
        "precision": m["test"]["macro_precision"], "recall": m["test"]["macro_recall"],
        "dev_macro_f1": m["dev"]["macro_f1"], "params": m.get("params"),
        "minutes": (m.get("train_seconds") or 0) / 60,
        "size_mb": m.get("model_size_mb"),
    } for m in metrics])
    df["is_best"] = df["macro_f1"] == df["macro_f1"].max()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Test macro-F1 across the ladder")
        base = alt.Chart(df).encode(
            x=alt.X("macro_f1:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".2f", title="test macro-F1")),
            y=alt.Y("model:N", sort=alt.EncodingSortField("rung"), title=None),
        )
        bar = base.mark_bar(height=22, cornerRadiusEnd=2).encode(
            color=alt.condition(alt.datum.is_best, alt.value(ACCENT), alt.value("#b9c4ce")),
            tooltip=["model", alt.Tooltip("macro_f1:Q", format=".4f", title="macro-F1"),
                     alt.Tooltip("accuracy:Q", format=".4f"),
                     alt.Tooltip("params:Q", format=",")])
        text = base.mark_text(align="left", dx=5, fontSize=11, color=INK).encode(
            text=alt.Text("macro_f1:Q", format=".3f"))
        st.altair_chart((bar + text).properties(height=34 * len(df)),
                        use_container_width=True)

    with right:
        st.markdown("#### Quality against cost")
        note("Up and to the left is better. The Transformer buys its accuracy with "
             "training time you can see on the log axis.")
        scat = (alt.Chart(df[df["minutes"] > 0]).mark_circle(size=170, opacity=0.85)
                .encode(
                    x=alt.X("minutes:Q", scale=alt.Scale(type="log"),
                            title="training minutes (log scale)"),
                    y=alt.Y("macro_f1:Q", scale=alt.Scale(zero=False),
                            title="test macro-F1"),
                    color=alt.Color("family:N", legend=alt.Legend(orient="bottom", title=None),
                                    scale=alt.Scale(scheme="tableau10")),
                    tooltip=["model", alt.Tooltip("macro_f1:Q", format=".4f"),
                             alt.Tooltip("minutes:Q", format=".1f"),
                             alt.Tooltip("size_mb:Q", title="size (MB)")])
                .properties(height=300))
        st.altair_chart(scat, use_container_width=True)

    st.markdown("#### Full comparison")
    show = df.drop(columns=["rung", "is_best"]).rename(columns={
        "model": "Model", "family": "Family", "macro_f1": "macro-F1",
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "dev_macro_f1": "Dev macro-F1", "params": "Params", "minutes": "Minutes",
        "size_mb": "Size (MB)"})
    if table is not None and "Δ vs previous (pp)" in table:
        show["Δ vs previous (pp)"] = table["Δ vs previous (pp)"].values
        show["Δ vs floor (pp)"] = table["Δ vs floor (pp)"].values
    st.dataframe(
        show.style
        .format({"macro-F1": "{:.4f}", "Accuracy": "{:.4f}", "Precision": "{:.4f}",
                 "Recall": "{:.4f}", "Dev macro-F1": "{:.4f}", "Minutes": "{:.1f}",
                 "Params": "{:,.0f}", "Size (MB)": "{:.1f}",
                 "Δ vs previous (pp)": "{:+.2f}", "Δ vs floor (pp)": "{:+.2f}"}, na_rep="—")
        .background_gradient(subset=["macro-F1"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### What each rung changed")
    for m in metrics:
        with st.expander(f"Rung {m['rung']} · {m['name']} — macro-F1 "
                         f"{m['test']['macro_f1']:.4f}"):
            st.write(C.MODEL_BY_KEY[m["key"]]["idea"])
            if m.get("config"):
                st.json(m["config"], expanded=False)

# ============================================================== DATASET ======
with tabs[1]:
    if not stats:
        st.info("Run `python src/01_data.py` to populate this tab.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: stat(f"{stats['n_classes']}", "intent classes")
        with c2: stat(f"{sum(stats['rows'].values()):,}", "labelled queries")
        with c3: stat(f"{stats['token_len']['mean']:.1f}", "mean tokens per query")
        with c4: stat(f"{stats['oov_rate_dev']:.1%}", "out-of-vocabulary rate on dev")

        st.markdown("<hr>", unsafe_allow_html=True)
        note(f"Splits are used strictly: train fits, dev validates and stops training, "
             f"test is scored once per rung. Class sizes vary by "
             f"{stats['imbalance_ratio']:.2f}× between the smallest and largest intent, "
             f"which is why macro-F1 rather than accuracy decides the ranking — macro-F1 "
             f"weights every intent equally, so a model cannot hide a weak class behind "
             f"a strong one. The out-of-vocabulary figure is the share of dev tokens the "
             f"word-level models can only see as &lt;OOV&gt;; DistilBERT's subword "
             f"tokenizer drives it to zero.")

        a, b = st.columns([2, 3], gap="large")
        with a:
            st.markdown("#### Split sizes")
            sp = pd.DataFrame({"split": list(stats["rows"]), "rows": list(stats["rows"].values())})
            st.altair_chart(
                alt.Chart(sp).mark_bar(color=ACCENT, cornerRadiusEnd=2).encode(
                    x=alt.X("rows:Q", title=None),
                    y=alt.Y("split:N", sort=["train", "dev", "test"], title=None),
                    tooltip=["split", "rows"]).properties(height=150),
                use_container_width=True)
        with b:
            st.markdown("#### Class distribution in train")
            cc = (pd.DataFrame({"intent": list(stats["class_counts"]),
                                "count": list(stats["class_counts"].values())})
                  .sort_values("count", ascending=False))
            st.altair_chart(
                alt.Chart(cc).mark_bar(color="#8aa9bd").encode(
                    x=alt.X("intent:N", sort="-y", axis=alt.Axis(labels=False, title=None)),
                    y=alt.Y("count:Q", title="training examples"),
                    tooltip=["intent", "count"]).properties(height=210),
                use_container_width=True)

        st.markdown("#### Sample queries")
        st.dataframe(pd.DataFrame(stats["examples"]), use_container_width=True,
                     hide_index=True)

# ============================================================= TRAINING ======
with tabs[2]:
    curved = [m for m in metrics if m.get("history")]
    if not curved:
        st.info("No training histories yet — run the recurrent rungs.")
    else:
        note("Training loss falling while dev loss turns upward is overfitting. Early "
             "stopping with restore_best_weights keeps the reported score at the dev "
             "minimum rather than the last epoch, so a long run is never rewarded for "
             "memorising the training set.")
        pick = st.multiselect("Models", [m["name"] for m in curved],
                              default=[m["name"] for m in curved])
        rows = []
        for m in curved:
            if m["name"] not in pick:
                continue
            h = m["history"]
            for i, v in enumerate(h.get("loss", []), 1):
                rows.append({"Model": m["name"], "epoch": i, "loss": v, "split": "train"})
            for i, v in enumerate(h.get("val_loss", []), 1):
                rows.append({"Model": m["name"], "epoch": i, "loss": v, "split": "dev"})
        curve_df = pd.DataFrame(rows).dropna()
        if not curve_df.empty:
            st.altair_chart(
                alt.Chart(curve_df).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("loss:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("split:N", scale=alt.Scale(
                        domain=["train", "dev"], range=[ACCENT, "#c9743a"]),
                        legend=alt.Legend(orient="top", title=None)),
                    facet=alt.Facet("Model:N", columns=3, title=None),
                    tooltip=["Model", "split", "epoch", alt.Tooltip("loss", format=".4f")]
                ).properties(width=260, height=190).resolve_scale(y="independent"),
                use_container_width=True)

        st.markdown("#### Epochs and stopping points")
        st.dataframe(pd.DataFrame([{
            "Model": m["name"], "Epochs run": m.get("epochs_run"),
            "Best epoch": m.get("best_epoch", "—"),
            "Train minutes": round((m.get("train_seconds") or 0) / 60, 1),
            "Params": m.get("params"),
        } for m in curved]), use_container_width=True, hide_index=True)

# =============================================================== ERRORS ======
with tabs[3]:
    if not analysis or not analysis.get("confused_pairs"):
        st.info("Run `python src/06_analysis.py` to populate this tab.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1: stat(analysis["best_name"], "model under the microscope")
        with c2: stat(f"{analysis['n_wrong']:,}", f"errors out of {analysis['n_test']:,}")
        with c3: stat(f"{analysis['error_rate']:.1%}", "test error rate")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### Where the errors concentrate")
        note("These pairs are semantically adjacent — a human annotator would hesitate "
             "on several of them. That means part of the remaining headroom is label "
             "ambiguity rather than model capacity, and the fix is intent definitions "
             "and targeted examples, not a bigger model.")

        pairs = pd.DataFrame([{"True intent": c["true"], "Predicted": c["predicted"],
                               "Errors": c["count"]} for c in analysis["confused_pairs"]])
        cc1, cc2 = st.columns([2, 3], gap="large")
        with cc1:
            st.altair_chart(
                alt.Chart(pairs.head(10)).mark_bar(color=ACCENT, cornerRadiusEnd=2).encode(
                    x=alt.X("Errors:Q", title=None),
                    y=alt.Y("True intent:N", sort="-x", title=None),
                    tooltip=["True intent", "Predicted", "Errors"]).properties(height=300),
                use_container_width=True)
        with cc2:
            st.dataframe(pairs, use_container_width=True, hide_index=True, height=300)

        st.markdown("#### Misclassified queries, in the model's own words")
        for c in analysis["confused_pairs"][:6]:
            with st.expander(f"{c['true']} → {c['predicted']} · {c['count']} errors"):
                for ex in c["examples"]:
                    st.markdown(f"- {ex}")

        w1, w2 = st.columns(2, gap="large")
        with w1:
            st.markdown("#### Weakest intents")
            st.dataframe(pd.DataFrame(analysis["weakest_classes"]).round(3),
                         use_container_width=True, hide_index=True)
        with w2:
            st.markdown("#### Strongest intents")
            st.dataframe(pd.DataFrame(analysis["strongest_classes"]).round(3),
                         use_container_width=True, hide_index=True)

        fig = C.FIG_DIR / "confusion_hard_classes.png"
        if fig.exists():
            st.markdown("#### Confusion among the 15 hardest classes")
            st.image(str(fig))

# =============================================================== TRY IT ======
with tabs[4]:
    import src.inference as inference

    ready = inference.available_models()
    if not ready:
        st.info("No saved models found yet.")
    else:
        st.markdown("#### Send one query through every rung")
        note("The same sentence, six different notions of what a sentence is. Watch the "
             "classical and recurrent models agree on easy queries and diverge on ones "
             "that need context or contain words outside their vocabulary.")

        examples = [
            "my card payment was declined at the shop",
            "how long does a transfer from the UK take to arrive",
            "I was charged a fee I don't recognise",
            "the app won't let me verify my identity",
            "why is my top-up still pending",
        ]
        pick_ex = st.selectbox("Start from an example, or type your own below",
                               ["—"] + examples)
        default = "" if pick_ex == "—" else pick_ex
        text = st.text_area("Customer query", value=default, height=90,
                            placeholder="Type a banking query…")

        chosen = st.multiselect(
            "Models to run", [m["key"] for m in ready],
            default=[m["key"] for m in ready],
            format_func=lambda k: C.MODEL_BY_KEY[k]["name"])

        if st.button("Classify", type="primary") and text.strip():
            cols = st.columns(min(3, max(1, len(chosen))))
            results = {}
            for i, key in enumerate(chosen):
                with cols[i % len(cols)]:
                    meta = C.MODEL_BY_KEY[key]
                    st.markdown(f"**{meta['name']}**")
                    try:
                        with st.spinner("…"):
                            r = inference.predict(key, text, top_k=3)
                        results[key] = r
                        top_label, top_p = r["top"][0]
                        st.markdown(f"<div class='stat'><div class='v' "
                                    f"style='font-size:1.15rem'>{top_label}</div>"
                                    f"<div class='k'>{top_p:.1%} confident · "
                                    f"{r['latency_ms']} ms</div></div>",
                                    unsafe_allow_html=True)
                        st.altair_chart(
                            alt.Chart(pd.DataFrame(r["top"], columns=["intent", "p"]))
                            .mark_bar(color=ACCENT, cornerRadiusEnd=2)
                            .encode(x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                                            axis=alt.Axis(format=".0%", title=None)),
                                    y=alt.Y("intent:N", sort="-x", title=None),
                                    tooltip=["intent", alt.Tooltip("p", format=".2%")])
                            .properties(height=110),
                            use_container_width=True)
                    except Exception as e:                       # noqa: BLE001
                        st.error(f"{type(e).__name__}: {e}")

            if len(results) > 1:
                tops = {k: v["top"][0][0] for k, v in results.items()}
                if len(set(tops.values())) == 1:
                    st.success(f"All {len(tops)} models agree: **{next(iter(tops.values()))}**")
                else:
                    st.warning("The models disagree — the kind of query worth adding to "
                               "the training set.")
                    st.dataframe(pd.DataFrame([
                        {"Model": C.MODEL_BY_KEY[k]["name"], "Prediction": v["top"][0][0],
                         "Confidence": f"{v['top'][0][1]:.1%}",
                         "Latency (ms)": v["latency_ms"]}
                        for k, v in results.items()]),
                        use_container_width=True, hide_index=True)
