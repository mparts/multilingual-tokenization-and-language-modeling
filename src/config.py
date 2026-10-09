"""
Config file holding paths and constants for the project.
"""

from pathlib import Path
import torch


# === Paths =======================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
VOCAB_DIR = DATA_DIR / "vocab"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = ROOT / "models"

INTERNAL_CORPUS_DIR = DATA_DIR / "corpus"
EXTERNAL_CORPUS_DIR = Path("/srv/data/lt2326-h26/a1")
# Uncomment this bellowe, and comment the above, in case not run in mlt gpu
# EXTERNAL_CORPUS_DIR = INTERNAL_CORPUS_DIR 

# === Constants ====================================================
LANGS = ["en", "tr", "zh"]
SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3


# === Tokenizing =========================================================
N_SAMPLES = 5


# === Training =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Tokenizer - Model names ==========================================
# List of available tokenizers (just for reference, variable isn't used anywhere, safe to alter):
all_toks     = ["Character-Level",
                "ByteLevelBPE_2000", "ByteLevelBPE_4000", "ByteLevelBPE_6000", "ByteLevelBPE_8000", "ByteLevelBPE_10000", 
                "ByteLevelBPE_12000", "ByteLevelBPE_14000", "ByteLevelBPE_16000", "ByteLevelBPE_18000", "ByteLevelBPE_20000", 
                "CharLevelBPE_2000", "CharLevelBPE_10000", "CharLevelBPE_11000", "CharLevelBPE_12000", "CharLevelBPE_15000", "CharLevelBPE_20000"]

# Keep one uncomented!!
MODELS = ["Character-Level", "ByteLevelBPE_2000", "ByteLevelBPE_10000", "ByteLevelBPE_20000", "CharLevelBPE_20000"]
# MODELS = ["Character-Level", "ByteLevelBPE_10000", "CharLevelBPE_20000"]
# MODELS = ["Character-Level",
#                 "ByteLevelBPE_2000", "ByteLevelBPE_4000", "ByteLevelBPE_6000", "ByteLevelBPE_8000", "ByteLevelBPE_10000", 
#                 "ByteLevelBPE_12000", "ByteLevelBPE_14000", "ByteLevelBPE_16000", "ByteLevelBPE_18000", "ByteLevelBPE_20000", 
#                 "CharLevelBPE_2000", "CharLevelBPE_10000", "CharLevelBPE_11000", "CharLevelBPE_12000", "CharLevelBPE_15000", "CharLevelBPE_20000"]
# MODELS = ["Character-Level", "CharLevelBPE_2000", "CharLevelBPE_10000", "ByteLevelBPE_2000", "ByteLevelBPE_10000"]
# MODELS = ["ByteLevelBPE_2000", "ByteLevelBPE_4000", "ByteLevelBPE_6000", "ByteLevelBPE_8000", "ByteLevelBPE_10000"]
# MODELS = ["CharLevelBPE_10000", "CharLevelBPE_11000", "CharLevelBPE_12000", "CharLevelBPE_15000", "CharLevelBPE_20000"]
# MODELS = ["ByteLevelBPE_10000", "ByteLevelBPE_12000", "ByteLevelBPE_14000", "ByteLevelBPE_16000", "ByteLevelBPE_18000", "ByteLevelBPE_20000"]