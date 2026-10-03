"""
Module holding the three tokenizers used in the assignment. Also, functions for reading data, building vocabularies, and computing statistics.
"""

from config import EXTERNAL_CORPUS_DIR, VOCAB_DIR, LOG_DIR
from config import LANGS, MODELS, SPECIALS, PAD, UNK, BOS, EOS
import json
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from datetime import datetime
import pprint as pp
from tqdm import tqdm


def read_lines(split, lang):
    with open(EXTERNAL_CORPUS_DIR / split / f"{lang}.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_vocab(sentences):
    counts = Counter()
    for s in tqdm(sentences, desc="Building character-level vocabulary"):
        counts.update(s)
    chars = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    itos = SPECIALS + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def encode(text, stoi):
    return [stoi.get(c, UNK) for c in text]


def ByteLevelBPE(vocab_size, load_path, save_path):
    """
    In my opinion most reasonable BPE tokenizer to use for multilingual data. ByteLevel makes sense.. 
    Not sure if assingment allows it though
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train([str(load_path)], trainer)
    tok.save(str(save_path))
    return tok


def BPE(vocab_size, load_path, save_path):
    """
    My second choice.. doesn't seem as good as the bytelevel but I think it is what the assingment actually wants us to do..
    I'm gonna send an email about this, but for the time being I am going to keep both approaches.
    """
    tok = Tokenizer(models.BPE(unk_token="<UNK>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tok.decoder = decoders.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        limit_alphabet=vocab_size - len(SPECIALS),
        show_progress=True,
    )

    tok.train([str(load_path)], trainer)
    tok.save(str(save_path))
    return tok


def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def stats(sentences, tokenized, mode):
    n_chars = sum(len(s) for s in sentences)
    if mode == "Character-Level":
        ids = [encode(s, tokenized) for s in sentences]
        n_unk = sum(id.count(UNK) for id in ids)
    else:
        ids = [tokenized.encode(s).ids for s in sentences]
        n_unk = sum(id.count(tokenized.token_to_id("<UNK>")) for id in ids)
    n_tokens = sum(len(id) for id in ids)
    return {"tokens": n_tokens, "tok/sent": f"{n_tokens / len(sentences):.2f}",
            "chars/tok": f"{n_chars / n_tokens:.2f}", "unk_percent": f"{100 * n_unk / n_tokens:.3f}%"}


if __name__ == "__main__":
    train_split = {l: read_lines("train", l) for l in LANGS} # get the training data for each language
    valid_split = {l: read_lines("valid", l) for l in LANGS} # get the validation data for each language
    balanced_split = EXTERNAL_CORPUS_DIR / "tokenizer" / "balanced.txt"
    statistics = {}

    for mode in MODELS:
        print("="*100)
        print(f"[{datetime.now()}] Training {mode} Tokenizer...", flush=True)
    
        if mode == "Character-Level":
            itos, stoi = build_vocab([s for l in LANGS for s in train_split[l]]) # build vocabulary
            statistics[mode] = {l: stats(valid_split[l], stoi, mode) for l in LANGS} # compute statistics for validation data
            statistics[mode]["vocab_size"] = len(itos) # add vocabulary size to statistics
            save_data(itos,VOCAB_DIR / f"{mode}_vocab.json") # save vocabulary

        elif mode.startswith("ByteLevelBPE"): # ByteLevel BPE
            size = 2000 if "2000" in mode else 10000 # set vocabulary size

            tok = ByteLevelBPE(size, balanced_split, f"{VOCAB_DIR}/{mode}_vocab.json") # train ByteLevel BPE tokenizer

            statistics[mode] = {l: stats(valid_split[l], tok, mode) for l in LANGS} # compute statistics for validation data
            statistics[mode]["vocab_size"] = tok.get_vocab_size() # add vocabulary size to statistics

        elif mode.startswith("BPE"): # BPE
            size = 2000 if "2000" in mode else 10000 # set vocabulary size

            tok = BPE(size, balanced_split, f"{VOCAB_DIR}/{mode}_vocab.json") # train BPE tokenizer

            statistics[mode] = {l: stats(valid_split[l], tok, mode) for l in LANGS} # compute statistics for validation data
            statistics[mode]["vocab_size"] = tok.get_vocab_size() # add vocabulary size to statistics

        print(f"\n[{datetime.now()}] Success!! Analysis of {mode} tokenizer on the validation split:", flush=True)
        print("Vocabulary size: ", statistics[mode]["vocab_size"])
        pp.pprint({x: statistics[mode][x] for x in statistics[mode] if x != "vocab_size"}, width=100)
        print("="*100)
    
    save_data(statistics, LOG_DIR / "tokenizer_stats.json") # save statistics
    print(f"\n[{datetime.now()}] All done! Tokenizer statistics saved to {LOG_DIR / 'tokenizer_stats.json'}", flush=True)


"""
Character-level tokenization yields a vocab arround 9k.. (because of chinese characters) This means that a regular BPE tokenizer 
with a vocab size of 2k will actually lose A LOT of information. Not even actually merge.. The 10k will just merge like, half a 
thousand characters and that's it. I'm not sure if I am misunderstanding the assignment.. The bytelevel bpe seems more reasonable, 
but not really similar to the bpe implementation of the link at the start of the assignment
"""

"""
The above comment + the docstrings for both bpe and bytelevel_bpe functions, were written before asking about it. After asking in 
class about bpe and how we should implement it, I realized that the bytelevel bpe is the right one. So I had slightly misunderstood
the purpose of the link in the assingment explaining the bpe algorithm. Either way, I leave the comments there just for documenting.
"""