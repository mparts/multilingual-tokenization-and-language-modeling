"""
helpers.py
"""

from config import EXTERNAL_CORPUS_DIR, UNK, VOCAB_DIR, MODELS
from tokenizers import Tokenizer
import json


def read_lines(split, lang, path=EXTERNAL_CORPUS_DIR):
    with open(path / split / f"{lang}.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def char_lvl_encode(text, stoi):
    return [stoi.get(c, UNK) for c in text]


def save_data(data, path, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "a":
        if path.exists() and path.stat().st_size > 0:
            with open(path, "r", encoding="utf-8") as f:
                runs = json.load(f)
        else:
            runs = []
        runs.append(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


def load_vocabulary(name, path=VOCAB_DIR):
    return Tokenizer.from_file(f"{path}/{name}_vocab.json")


def load_tokenizer(name):
    if name == MODELS[0]:
        itos = json.load(open(VOCAB_DIR / f"{name}_vocab.json", encoding="utf-8"))
        stoi = {c: i for i, c in enumerate(itos)}
        return (lambda ss: [char_lvl_encode(s, stoi) for s in ss]), len(itos)
    tok = load_vocabulary(name)
    return (lambda ss: [e.ids for e in tok.encode_batch(ss)]), tok.get_vocab_size()