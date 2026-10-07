from config import LOG_DIR, MODELS, LANGS
from helpers import load_json
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from datetime import datetime


def learning_curves(out_dir):
    runs = {t: load_json(LOG_DIR / "train_run" / f"{t}_train.json") for t in MODELS}
    runs = {t: r for t, r in runs.items() if r}
    if not runs:
        print(f"No files found for train.")
        return
    fig, (ax_loss, ax_bpc) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (tok, run) in enumerate(runs.items()):
        ep = run["history"]["epochs"]
        x = [e["epoch"] for e in ep]
        c = f"C{i}"
        ax_loss.plot(x, [e["train_loss"] for e in ep], "--", color=c, label=f"{tok} train")
        ax_loss.plot(x, [e["valid_loss"] for e in ep], "-", color=c, label=f"{tok} valid")
        ax_bpc.plot(x, [e["valid_bpc_approx"] for e in ep], "-o", color=c, label=tok)

    ax_loss.set(xlabel="epoch", ylabel="cross-entropy (nats/token)", title="Train vs. valid loss")
    ax_bpc.set(xlabel="epoch", ylabel="bits per character (approx.)", title="Valid BPC (comparable across tokenizers)")
    for ax in (ax_loss, ax_bpc):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def eval_plots(split, out_dir):
    res = {t: load_json(LOG_DIR / f"{split}_run" / f"{t}_{split}_eval.json") for t in MODELS}
    res = {t: r["results"] for t, r in res.items() if r}
    if not res:
        print(f"No eval files found for split '{split}'.")
        return
    groups = ["all"] + list(LANGS)
    w = 0.8 / len(res)
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.9), 4.8))
    for i, (tok, r) in enumerate(res.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [r[g]["bpc_approx"] for g in groups], w, label=tok)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set(ylabel="bits per character (approx.)", title=f"{split} BPC per language")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_bpc_per_language.png", dpi=150)
    plt.close(fig)

    langs = list(LANGS)
    mat = np.array([[res[t][l]["bpc_approx"] for l in langs] for t in res])
    fig, ax = plt.subplots(figsize=(max(7, len(langs) * 0.7), 1 + 0.6 * len(res)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs)
    ax.set_yticks(range(len(res)))
    ax.set_yticklabels(list(res))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, label="BPC")
    ax.set_title(f"{split} BPC heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_bpc_heatmap.png", dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    args = ap.parse_args()
    out_dir = LOG_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] Generating plots for '{args.split}' split...", flush=True)
    learning_curves(out_dir)
    eval_plots(args.split, out_dir)
    print(f"Plots saved to {out_dir}")
