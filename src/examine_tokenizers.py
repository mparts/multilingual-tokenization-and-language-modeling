"""
Simple inspector for the tokenizers. It will print out the number of distinct tokens used in the validation set, 
and then print out a few random examples of tokenization for each language and tokenizer.
"""


import json
from pathlib import Path
from tokenizers import Tokenizer
import random

DATA = Path("/srv/data/lt2326-h26/a1")
VOCAB_PATH = Path("../data/vocab")
SAVE_PATH = Path("../data/logs")
LANGS = ["en", "tr", "zh"]
MODELS = ["BPE_2000", "BPE_10000", "ByteLevelBPE_2000", "ByteLevelBPE_10000"]


def read_lines(split, lang):
    with open(DATA / split / f"{lang}.txt", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def pieces(tok, text):
    return [tok.decode([i]) for i in tok.encode(text).ids]


def save_data(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


valid = {l: read_lines("valid", l) for l in LANGS}
toks = {n: Tokenizer.from_file(f"{VOCAB_PATH}/{n}_vocab.json") for n in MODELS}


data = {}
for l in LANGS:
    data[l] = {}
    for text in (random.sample(valid[l], 1)):
        print("\n", "="*100, "\n", text)
        print("\nchar :", list(text))
        data[l][text] = {"char": ", ".join(list(text))}
        for n, tok in toks.items():
            print(f"\n{n:>5}:", pieces(tok, text))
            data[l][text][n] = ", ".join(pieces(tok, text))

save_data(data, SAVE_PATH / "tokenized_sentences.json")