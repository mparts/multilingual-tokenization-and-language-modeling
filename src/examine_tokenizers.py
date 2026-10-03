"""
Simple inspector for the tokenizers. Prints out a few random examples of tokenization for each language and tokenizer.
"""

from config import LOG_DIR
from config import LANGS, MODELS
from datetime import datetime
import random
from helpers import read_lines, save_data, load_vocabulary

def pieces(tok, text):
    return [tok.decode([i]) for i in tok.encode(text).ids]


valid = {l: read_lines("valid", l) for l in LANGS}
toks = {n: load_vocabulary(n) for n in MODELS[1:]}


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

save_data(data, LOG_DIR / "tokenizers" / "tokenized_sentences.json")
print(f"\n[{datetime.now()}] All done! Tokenized sentences exported to {LOG_DIR / 'tokenizers' / 'tokenized_sentences.json'}", flush=True)