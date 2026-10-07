"""
Simple inspector for the tokenizers. It will print out the number of distinct tokens used in the validation set, 
and then print out a few random examples of tokenization for each language and tokenizer.
"""


from pathlib import Path
from tokenizers import Tokenizer
import random
import pandas as pd
from datetime import datetime

DATA = Path("/srv/data/lt2326-h26/a1")
LANGS = ["en", "tr", "zh"]


def read_lines(split, lang):
    with open(DATA / split / f"{lang}.txt", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def pieces(tok, text):
    return [tok.decode([i]) for i in tok.encode(text).ids]


print(f"[{datetime.now()}] Inspecting tokenization...\n", flush=True)
valid = {l: read_lines("valid", l) for l in LANGS}
toks = {n: Tokenizer.from_file(f"../data/vocab/{n}_vocab.json") for n in ("BPE_2000", "BPE_10000", "ByteLevelBPE_2000", "ByteLevelBPE_10000")}


data = []
for n, tok in toks.items():
    for l in LANGS:
        used = {i for s in valid[l] for i in tok.encode(s).ids}
        data.append({
            "Tokenizer": n,
            "Language": l,
            "Distinct Tokens": len(used)
        })
df = pd.DataFrame(data)
print("Distinct token types used on validation:")
print(df.to_string(index=False))


for l in LANGS:
    for text in (random.sample(valid[l], 3)):
        print("\n", "="*100, "\n", text)
        print("\nchar :", list(text))
        for n, tok in toks.items():
            print(f"\n{n:>5}:", pieces(tok, text))