"""
FWC Module 5 · Test-6 — central configuration.

Every path, seed and hyperparameter lives here so the report can quote the
exact settings that produced the numbers.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"                 # you drop train.xlsx / dev.xlsx / test.xlsx here
ART = ROOT / "artifacts"
PROC_DIR = ART / "processed"             # cleaned CSVs, label list, vocabulary
MODEL_DIR = ART / "models"               # saved models, one per rung
METRIC_DIR = ART / "metrics"             # one JSON per rung
PRED_DIR = ART / "predictions"           # test-set predictions for error analysis
FIG_DIR = ART / "figures"                # PNGs for the report
GLOVE_DIR = ART / "glove"                # cached embedding file
REPORT_PATH = ROOT / "REPORT.md"
COMPARISON_CSV = ART / "comparison.csv"
ANALYSIS_JSON = ART / "analysis.json"
DATASET_STATS = PROC_DIR / "dataset_stats.json"

for _d in (DATA_DIR, ART, PROC_DIR, MODEL_DIR, METRIC_DIR, PRED_DIR, FIG_DIR, GLOVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Text / vocabulary (shared by every Keras rung so the comparison is fair)
# ----------------------------------------------------------------------------
MAX_VOCAB = 20_000
MAX_LEN = 40          # Banking77 queries are short; 01_data.py prints the percentiles
EMBED_DIM = 100       # must match glove.6B.100d

# ----------------------------------------------------------------------------
# Keras recurrent rungs
# ----------------------------------------------------------------------------
RNN_UNITS = 64
BATCH_SIZE = 64
EPOCHS = 30           # early stopping decides the real number
PATIENCE = 4
LR = 1e-3

# Enhanced rung (BiLSTM + GloVe)
GLOVE_NAME = "glove-wiki-gigaword-100"          # gensim id (~128 MB download)
GLOVE_FALLBACK_URL = "https://nlp.stanford.edu/data/glove.6B.zip"   # 822 MB fallback
GLOVE_TRAINABLE = True     # start from GloVe, keep fine-tuning on our corpus
BILSTM_UNITS = 96
BILSTM_DROPOUT = 0.3
BILSTM_DENSE_DROPOUT = 0.4

# ----------------------------------------------------------------------------
# Transfer-learning rung (DistilBERT)
# ----------------------------------------------------------------------------
HF_MODEL = "distilbert-base-uncased"
BERT_MAX_LEN = 64
BERT_BATCH = 16
BERT_EVAL_BATCH = 64
BERT_EPOCHS = 4
BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.01

# ----------------------------------------------------------------------------
# The ladder. Order here == order everywhere (tables, charts, report).
# ----------------------------------------------------------------------------
MODELS = [
    {"key": "tfidf_lr",     "name": "TF-IDF + LogisticRegression",
     "family": "Classical floor",       "rung": 0,
     "idea": "Sparse frequency features, no word order, no semantics. The number every neural model must beat."},
    {"key": "simple_rnn",   "name": "SimpleRNN",
     "family": "Recurrent",             "rung": 1,
     "idea": "Adds memory and word order, but BPTT gradients vanish over long spans."},
    {"key": "gru",          "name": "GRU",
     "family": "Recurrent",             "rung": 2,
     "idea": "Two gates (update, reset) give the network a gradient highway with few extra parameters."},
    {"key": "lstm",         "name": "LSTM",
     "family": "Recurrent",             "rung": 3,
     "idea": "Three gates plus an additively-updated cell state — the classic long-range memory fix."},
    {"key": "bilstm_glove", "name": "BiLSTM + GloVe (enhanced)",
     "family": "Recurrent (enhanced)",  "rung": 4,
     "idea": "Same LSTM, four enhancements: bidirectional reading, pre-trained GloVe init, dropout, LR scheduling."},
    {"key": "distilbert",   "name": "DistilBERT (fine-tuned)",
     "family": "Transfer learning",     "rung": 5,
     "idea": "Pre-trained bidirectional Transformer encoder + a classification head on [CLS]. Contextual embeddings replace static ones."},
]

MODEL_BY_KEY = {m["key"]: m for m in MODELS}
