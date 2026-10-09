"""
helpers.py
holds various helper functions
"""

from config import EXTERNAL_CORPUS_DIR, UNK, VOCAB_DIR, ROOT
from tokenizers import Tokenizer
import json


def read_lines(split, lang, path=EXTERNAL_CORPUS_DIR):
    """
    simply reads from the corpus file
    """
    with open(path / split / f"{lang}.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def char_lvl_encode(text, stoi):
    """
    encodes the Character-level vocab
    """
    return [stoi.get(c, UNK) for c in text]


def load_json(path):
    """
    simply loads a .json
    """
    if not path.exists():
        print(f"  [!] missing: {path.relative_to(ROOT)}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path, mode="w"):
    """
    saves stuff into a .json. (I am probably calling this over a gazillion times)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "a":
        if path.exists() and path.stat().st_size > 0:
            runs = load_json(path)
        else:
            runs = []
        runs.append(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_vocabulary(name, path=VOCAB_DIR):
    """
    loads a hugging face tokenizer
    """
    return Tokenizer.from_file(f"{path}/{name}_vocab.json")


def load_tokenizer(name):
    """
    loads the tokenizers, encodes, and also returns their size alongside them
    """
    if name == "Character-Level":
        itos = load_json(VOCAB_DIR / f"{name}_vocab.json")
        stoi = {c: i for i, c in enumerate(itos)}
        return (lambda ss: [char_lvl_encode(s, stoi) for s in ss]), len(itos)
    tok = load_vocabulary(name)
    return (lambda ss: [e.ids for e in tok.encode_batch(ss)]), tok.get_vocab_size()