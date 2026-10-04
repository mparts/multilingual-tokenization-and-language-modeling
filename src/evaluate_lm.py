"""
evaluate_lm.py
"""

from config import MODEL_DIR, MODELS, LANGS, LOG_DIR, DEVICE, RUN_HISTORY, LATEST
from helpers import save_data, load_tokenizer
from language_model import make_blocks, evaluate, bpc_approx, load_checkpoint

import math
from datetime import datetime
from argparse import ArgumentParser


def main():
    argparser = ArgumentParser(description="Evaluates a trained Transformer language model on a corpus split.")
    argparser.add_argument("--tok",        type=str, required=True, choices=MODELS, help="Tokenizer whose model to evaluate.")
    argparser.add_argument("--split",      type=str, default="test", choices=["valid", "test"], help="Split to evaluate on (default: test).")
    argparser.add_argument("--batch_size", type=int, default=64, help="Evaluation batch size (default: 64).")
    args = argparser.parse_args()

    print("="*100)
    print(f"[{datetime.now()}] Loading trained model...", flush=True)
    model, _, meta = load_checkpoint(MODEL_DIR / f"{args.tok}.pt", DEVICE)
    encode, _ = load_tokenizer(args.tok)
    ctx = model.config["context_length"]

    print(f"\n[{datetime.now()}] Evaluating '{args.tok}' model on the '{args.split}' split...", flush=True)
    results = {}
    for name, langs in [("all", LANGS)] + [(l, [l]) for l in LANGS]:
        x, y, n_chars = make_blocks(encode, args.split, ctx, langs=langs)
        loss = evaluate(model, x, y, DEVICE, args.batch_size)
        results[name] = {"loss": loss, "perplexity": math.exp(loss),
                         "bpc_approx": bpc_approx(loss, y.numel() / n_chars), "blocks": len(x)}
        print(f"    {name:>3}: Loss: {results[name]['loss']:.3f} | "
              f"Perplexity: {results[name]['perplexity']:.3f} | BPC approx: {results[name]['bpc_approx']:.3f}")

    save_path, file_name = LOG_DIR / f"{args.split}_run", f"{args.tok}_{args.split}_eval.json"
    save_data({f"{datetime.now()}": {"split": args.split, **meta, "results": results}},
              save_path / RUN_HISTORY / file_name, mode="a")
    save_data({"split": args.split, **meta, "results": results},
              save_path / LATEST / file_name)
    print(f"\n[{datetime.now()}] Evaluation completed!!", flush=True)
    print("="*100)


if __name__ == "__main__":
    main()
