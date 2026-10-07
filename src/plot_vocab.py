import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import LANGS, LOG_DIR
from helpers import load_json

LANGUAGE_COLORS = {"en": "green", "tr": "red", "zh": "blue",}
FAMILY_MARKERS = {"ByteLevelBPE": "s", "CharLevelBPE": "o", "Character-Level": "^",}


def get_family(tokenizer):
    if tokenizer == "Character-Level":
        return "Character-Level"
    if tokenizer.startswith("ByteLevel"):
        return "ByteLevelBPE"
    return "CharLevelBPE"


def plot_vocab_stats(out_path, mode):
    if mode == "tok/sent":
        x_label = "Tokens per sentence (tok/sent)"
        sub_title = "Vocabulary Size vs. Tokens per Sentence"
        save_to = "toks-per-sent_VS_vocab.png"
        legend_loc = "upper right"
    elif mode == "chars/tok":
        x_label = "Characters per token (chars/tok)"
        sub_title = "Vocabulary Size vs. Characters per token"
        save_to = "chars-per-token_VS_vocab.png"
        legend_loc = "lower right"
    elif mode == "tokens":
        x_label = "Total Tokens"
        sub_title = "Vocabulary Size vs. Total Tokens"
        save_to = "tokens_VS_vocab.png"
        legend_loc = "upper right"


    fig, (ax_byte, ax_char) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True,)
    for lang in LANGS:
        byte_points, char_points = [], []

        for tokenizer, info in data.items():
            family = get_family(tokenizer)
            x = float(info[lang][mode])
            y = info["vocab_size"]
            point = (x, y, family)
            if family == "ByteLevelBPE":
                byte_points.append(point)
            else:
                char_points.append(point)

        byte_points.sort(key=lambda p: p[0])
        char_points.sort(key=lambda p: p[0])
        for ax, points in ((ax_byte, byte_points), (ax_char, char_points)):
            if not points:
                continue
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                color=LANGUAGE_COLORS[lang],
                linewidth=2, alpha=0.8, zorder=2,
            )
            for x, y, family in points:
                ax.plot(x, y,
                    marker=FAMILY_MARKERS[family],
                    markerfacecolor=LANGUAGE_COLORS[lang],
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=1.2, linestyle="None", zorder=3,
                )
    titles = ["ByteLevel Tokenizers", "CharLevelBPEs + Character-Level Tokenizer"]
    for ax, title in ((ax_byte, titles[0]), (ax_char, titles[1]),):
        ax.set_title(title)
        ax.set_ylabel("Vocabulary size")
        ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax_char.set_xlabel(x_label)

    language_labels = {
        "en": "English (en)",
        "tr": "Turkish (tr)",
        "zh": "Chinese (zh)",
    }
    language_legend = [
        Line2D([0], [0],
            color=LANGUAGE_COLORS[lang],
            linewidth=2,
            label=language_labels[lang],
        )
        for lang in LANGS
    ]
    family_legend = [
        Line2D([0], [0],
            marker=FAMILY_MARKERS[family],
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            linestyle="None",
            label=family,
        )
        for family in FAMILY_MARKERS
    ]
    ax_byte.legend(handles=language_legend + family_legend, loc=legend_loc,)
    fig.suptitle(sub_title)
    fig.tight_layout()
    fig.savefig(out_path / save_to, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)



if __name__ == "__main__":
    indir = LOG_DIR / "tokenizers"
    outdir = LOG_DIR / "plots"
    data = load_json(indir / "tokenizer_stats.json")
    plot_vocab_stats(outdir, "tok/sent")
    plot_vocab_stats(outdir, "chars/tok")
    plot_vocab_stats(outdir, "tokens")