"""
Config file holding paths and constants for the project.
"""

from pathlib import Path
import torch


# === Paths =======================================================
EXTERNAL_CORPUS_DIR = Path("/srv/data/lt2326-h26/a1")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
VOCAB_DIR = DATA_DIR / "vocab"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = ROOT / "models"
RUN_HISTORY = "run_history"
LATEST = "latest_runs"


# === Constants ====================================================
LANGS = ["en", "tr", "zh"]
MODELS = ["Character-Level", "CharLevelBPE_2000", "CharLevelBPE_10000", "ByteLevelBPE_2000", "ByteLevelBPE_10000"]
# MODELS = ["ByteLevelBPE_2000", "ByteLevelBPE_4000", "ByteLevelBPE_6000", "ByteLevelBPE_8000", "ByteLevelBPE_10000"]
# MODELS = ["CharLevelBPE_10000", "CharLevelBPE_11000", "CharLevelBPE_12000", "CharLevelBPE_15000", "CharLevelBPE_20000"]
SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# === Training =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"