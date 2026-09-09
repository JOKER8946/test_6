# Test-6 · Banking77 model ladder

Six models on one dataset, each rung changing exactly one idea from Module 5:
TF-IDF floor → SimpleRNN → GRU → LSTM → BiLSTM+GloVe → fine-tuned DistilBERT.
The pipeline writes a comparison table, figures, an auto-generated report and a
Streamlit dashboard with live prediction across all six models.

## 1. Put the data in place

Copy your Kaggle Banking77 files into `data/`:

```
data/train.xlsx
data/dev.xlsx
data/test.xlsx
```

Column names don't matter — `01_data.py` detects the text and label columns
whatever the export called them.

## 2. Create the environment

**Windows (PowerShell)**

```powershell
cd test6-banking77
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd test6-banking77
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code: `Ctrl/Cmd+Shift+P` → *Python: Select Interpreter* → pick `.venv`.
Then every script runs with the ▶ button.

Python 3.11 is the safe choice. 3.12 works but gensim wheels are occasionally
late; 3.13 is too new for this pin set.

## 3. Run the pipeline

In order, from the project root:

```bash
python src/01_data.py          # seconds  — clean, encode, vocabulary, stats
python src/02_tfidf.py         # ~1 min   — rung 0, the floor
python src/03_recurrent.py     # ~20-40 min — rungs 1-3 (SimpleRNN, GRU, LSTM)
python src/04_bilstm_glove.py  # ~15-30 min — rung 4 (+ a one-time GloVe download)
python src/05_distilbert.py    # hours    — rung 5, the slow one
python src/06_analysis.py      # seconds  — table, figures, error analysis
python src/07_report.py        # seconds  — REPORT.md from the real numbers
streamlit run app.py           # the dashboard
```

Timings are rough for a laptop CPU. Every rung caches its result, so re-running
a script skips work that is already done — a crash in stage 5 costs you nothing
from stages 1-4. To retrain something deliberately:

```bash
FORCE=1 python src/03_recurrent.py          # macOS / Linux
$env:FORCE=1; python src/03_recurrent.py    # PowerShell
```

or just delete the matching file in `artifacts/metrics/`.

You can run `06` and `07` at any point — they use whatever rungs have finished,
so you can watch the table fill in rather than waiting for DistilBERT.

## 4. What gets produced

```
artifacts/
  processed/     cleaned splits, labels.json, vocab.json, dataset_stats.json
  models/        one saved model per rung
  metrics/       one JSON per rung: dev + test metrics, timing, size, curves
  predictions/   test-set predictions, for error analysis
  figures/       macro_f1_ladder.png, training_curves.png, confusion_*.png
  comparison.csv the table
  analysis.json  confused pairs, weakest classes, error examples
REPORT.md        the write-up, regenerated from artifacts on every run of 07
```

For a PDF: open `REPORT.md` in VS Code with the *Markdown PDF* extension, or
`pandoc REPORT.md -o REPORT.pdf`.

## 5. Design decisions worth defending in the viva

**Why macro-F1 and not accuracy.** 77 classes of unequal size; accuracy lets a
model coast on the big ones. Macro-F1 weights every intent equally.

**Why dev exists.** Early stopping, the LR schedule and model selection all read
dev. Test is scored once per rung. If you tune on test, your reported number is
optimistically biased and an examiner will ask about it.

**Why preprocessing is minimal.** No stopword removal, no lemmatisation. Intent
detection turns on the exact words a stopword list deletes — *card not working*
and *card working* are different intents. This is the module's own warning about
negations, applied.

**Why the recurrent rungs share a vocabulary and hyperparameters.** Only the
cell changes between rungs 1-3, so the delta in the table is caused by gating,
not by one model getting a better configuration.

**Why DistilBERT sees raw text while the RNNs see cleaned text.** WordPiece was
pre-trained on natural text including punctuation; stripping it would move the
input away from the distribution the checkpoint expects. The RNNs have no such
prior and benefit from a smaller, denser vocabulary.

**Why the enhancement rung bundles four changes.** Test-6 asks for a baseline
and an enhancement, not an ablation. If an examiner asks which change mattered,
the honest answer is that GloVe initialisation is doing most of the work — with
only a few thousand training sentences there is not enough signal to learn good
embeddings from scratch — and you can prove it by setting
`GLOVE_TRAINABLE = False` or removing `embedding_matrix` and re-running.

**Where this goes next.** The winning model serves behind FastAPI in a container
(Module 2 §8) with drift monitoring. If the latency budget won't take it, distil
or quantise the Transformer rather than falling back to an LSTM — that is the
DistilBERT / MobileBERT / TinyBERT trade-off from §6.2.

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `FileNotFoundError: could not find a 'train' file` | The xlsx files aren't in `data/`, or are nested in a subfolder. |
| `ModuleNotFoundError: openpyxl` | The venv isn't active, or `pip install -r requirements.txt` didn't finish. |
| GloVe download stalls | It falls back to the 822 MB Stanford zip automatically. To skip it: download `glove.6B.zip`, put it in `artifacts/glove/`, re-run stage 4. |
| DistilBERT is unbearably slow | Lower `BERT_EPOCHS` to 2 or `BERT_MAX_LEN` to 48 in `src/config.py`, then `FORCE=1`. Note the change in the report. |
| TensorFlow prints oneDNN / CPU warnings | Harmless. They're informational on CPU builds. |
| Dashboard says "No results yet" | Run stages 1-2 at minimum; the dashboard renders whatever exists. |
