"""
Who gets the vocabulary?
Method
  rate_l(t)  = count of token t in language l / number of characters in language l
  share_l(t) = rate_l(t) / sum_l' rate_l'(t)
  token is "<l>"    if share_l(t) >= THRESH (default 0.8)
  token is "shared" if no language reaches THRESH
  token is "rare"   if it occurs fewer than MIN_COUNT times in total (incl. never used); special tokens are skipped
Rates are per character (not per token) because the three corpora have about equal characters but very different token counts.
Character-Level tokenizer always ignored.
"""

from config import LOG_DIR, LANGS, MODELS
from helpers import read_lines, load_vocabulary, save_data, load_json

from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt

THRESH = 0.9
MIN_COUNT = 5
N_SPECIAL = 4
CLASSES = LANGS + ["shared", "rare"]


def count_tokens(tok, sents_by_lang):
    counts = {l: Counter() for l in LANGS}
    for l in LANGS:
        for e in tok.encode_batch(sents_by_lang[l]):
            counts[l].update(e.ids)
    return counts


def classify(counts, n_chars, vocab_size, thresh=THRESH, min_count=MIN_COUNT):
    labels = {}
    for i in range(N_SPECIAL, vocab_size):
        c = {l: counts[l][i] for l in LANGS}
        if sum(c.values()) < min_count:
            labels[i] = "rare"
            continue
        rate = {l: c[l] / n_chars[l] for l in LANGS}
        top = max(rate, key=rate.get)
        labels[i] = top if rate[top] / sum(rate.values()) >= thresh else "shared"
    return labels


def analyse(tok, counts, n_chars, n_examples=10):
    V = tok.get_vocab_size()
    labels = classify(counts, n_chars, V)
    total_c = {l: sum(counts[l].values()) for l in LANGS}
    text = {i: tok.decode([i]) for i in labels}
    frag = {i for i, s in text.items() if "\ufffd" in s}          # partial UTF-8 byte pieces

    out = {"vocab_size": V}
    # 1) how many vocabulary entries each class gets
    out["vocab_entries"] = {c: sum(1 for v in labels.values() if v == c) for c in CLASSES}
    # 2) what share of each language's token stream is covered by each class
    out["stream_share"] = {l: {c: sum(counts[l][i] for i, v in labels.items() if v == c) / total_c[l]
                               for c in CLASSES} for l in LANGS}
    # 3) byte fragments (not valid characters on their own): count and which class owns them
    out["byte_fragments"] = {c: sum(1 for i in frag if labels[i] == c) for c in CLASSES}
    out["byte_fragment_stream_share"] = {l: sum(counts[l][i] for i in frag) / total_c[l] for l in LANGS}
    # 4) overlap between languages: tokens used >= MIN_COUNT times in both
    used = {l: {i for i in labels if counts[l][i] >= MIN_COUNT} for l in LANGS}
    out["overlap_jaccard"] = {f"{a}-{b}": len(used[a] & used[b]) / len(used[a] | used[b])
                              for a, b in [("en", "tr"), ("en", "zh"), ("tr", "zh")]}
    out["tokens_used"] = {l: len(used[l]) for l in LANGS}
    # 5) examples: most frequent tokens per class (frequency = per 1000 characters, all languages summed)
    ex = {}
    for c in CLASSES[:-1]:
        ids = sorted((i for i, v in labels.items() if v == c),
                     key=lambda i: -sum(counts[l][i] / n_chars[l] for l in LANGS))
        ex[c] = [text[i] for i in ids[:n_examples]]
    out["examples"] = ex
    return out


def plot_vocab_allocation(out_path):
    COLORS = {"en": "green", "tr": "red", "zh": "blue", "shared": "gray", "rare": "lightgray"}
    LABELS = {"en": "English", "tr": "Turkish", "zh": "Chinese", "shared": "Shared", "rare": "Rare (<5 uses)"}

    data = load_json(LOG_DIR / "tokenizers" / "vocab_allocation.json")
    names = list(data)
    fig, ax = plt.subplots(figsize=(9, 5))

    left = [0.0] * len(names)
    for cls in COLORS:
        pct = [100 * data[n]["vocab_entries"][cls] / data[n]["vocab_size"] for n in names]
        ax.barh(names, pct, left=left, color=COLORS[cls], edgecolor="white",
                hatch="//" if cls == "rare" else None, label=LABELS[cls])
        for i, p in enumerate(pct):
            if p >= 4:
                ax.text(left[i] + p / 2, i, f"{p:.0f}%", ha="center", va="center",
                        color="white" if cls != "rare" else "black", fontsize=9)
        left = [l + p for l, p in zip(left, pct)]

    ax.set(xlabel="Share of vocabulary entries (%)", xlim=(0, 100),
           title=f"Who gets the vocabulary? (threshold {THRESH}, min count {MIN_COUNT})")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / "vocab_allocation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path / 'vocab_allocation.png'}")



if __name__ == "__main__":
    train = {l: read_lines("train", l) for l in LANGS}
    n_chars = {l: sum(len(s) for s in train[l]) for l in LANGS}
    results = {}
    for name in MODELS:
        if name == "Character-Level":
            continue
        print(f"[{datetime.now():%H:%M:%S}] {name}", flush=True)
        tok = load_vocabulary(name)
        results[name] = analyse(tok, count_tokens(tok, train), n_chars)

    for name, r in results.items():
        print(f"\n=== {name} (threshold {THRESH}, min count {MIN_COUNT}) ===")
        print("vocab entries :", {c: f"{n} ({n / r['vocab_size']:.0%})" for c, n in r["vocab_entries"].items()})
        for l in LANGS:
            print(f"{l} stream     :", {c: f"{p:.0%}" for c, p in r["stream_share"][l].items()})
        print("byte fragments:", r["byte_fragments"], "| share of stream:",
              {l: f"{p:.0%}" for l, p in r["byte_fragment_stream_share"].items()})
        print("Jaccard       :", {k: round(v, 3) for k, v in r["overlap_jaccard"].items()}, "| tokens used:", r["tokens_used"])
        for c, e in r["examples"].items():
            print(f"top {c:6s}:", e)

    save_data(results, LOG_DIR / "tokenizers" / "vocab_allocation.json")
    plot_vocab_allocation( LOG_DIR / "plots")