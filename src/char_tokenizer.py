import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import pprint as pp

DATA_PATH = Path("/srv/data/lt2326-h26/a1")
# DATA_PATH = Path("../data/corpus")
VOCAB_PATH = Path("../data/vocab")
LANGS = ["en", "tr", "zh"]
SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3


def read_lines(split, lang):
    with open(DATA_PATH / split / f"{lang}.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_vocab(sentences):
    counts = Counter()
    for s in sentences:
        counts.update(s)
    chars = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    itos = SPECIALS + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def save_data(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def encode(text, stoi):
    return [stoi.get(c, UNK) for c in text]


def stats(sentences, stoi):
    n_chars = sum(len(s) for s in sentences)
    ids = [encode(s, stoi) for s in sentences]
    n_tokens = sum(len(x) for x in ids)
    n_unk = sum(x.count(UNK) for x in ids)
    return {"tokens": n_tokens, "tok/sent": f"{n_tokens / len(sentences):.2f}",
            "chars/tok": f"{n_chars / n_tokens:.2f}", "unk percent": f"{100 * n_unk / n_tokens:.3f}%"}


if __name__ == "__main__":
    print("="*100)
    print(f"[{datetime.now()}] Loading data and tokenizing in character level...", flush=True)
    train_split = {l: read_lines("train", l) for l in LANGS} # get the training data for each language
    itos, stoi = build_vocab([s for l in LANGS for s in train_split[l]]) # build vocabulary
    save_data(itos, VOCAB_PATH / "char_vocab.json") # save vocabulary to file
    statistics = {l: stats(read_lines("valid", l), stoi) for l in LANGS} # compute statistics for validation data
    save_data(statistics, VOCAB_PATH / "char_stats.json") # save statistics to file
    print(f"[{datetime.now()}] Data tokenized!! Statistics of the tokenizer on the validation split:\n", flush=True)
    print("vocab size:", len(itos))
    pp.pprint(statistics)
    print("="*100)