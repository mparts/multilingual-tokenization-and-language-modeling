"""
Simple inspector for the tokenizers. Prints out a few random examples of tokenization for each language and tokenizer.
"""

from config import EXTERNAL_CORPUS_DIR, VOCAB_DIR, LOG_DIR
from config import LANGS, MODELS
import json
from tokenizers import Tokenizer
import random


def read_lines(split, lang):
    with open(EXTERNAL_CORPUS_DIR / split / f"{lang}.txt", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def pieces(tok, text):
    return [tok.decode([i]) for i in tok.encode(text).ids]


def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


valid = {l: read_lines("valid", l) for l in LANGS}
toks = {n: Tokenizer.from_file(f"{VOCAB_DIR}/{n}_vocab.json") for n in MODELS}


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

save_data(data, LOG_DIR / "tokenized_sentences.json")