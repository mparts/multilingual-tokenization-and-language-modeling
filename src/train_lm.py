"""
train_lm.py
"""

from config import LOG_DIR, MODELS, DEVICE
from helpers import save_data, load_tokenizer
from language_model import TransformerLM, make_blocks, train

import torch
from datetime import datetime
from argparse import ArgumentParser


def main():
    argparser = ArgumentParser(description="Trains a Transformer language model on a tokenized multilingual corpus.")
    argparser.add_argument("--tok",            type=str,   required=True, choices=MODELS, help="Tokenizer to train on.")
    argparser.add_argument("--epochs",         type=int,   default=10,    help="Number of training epochs (default: 10).")
    argparser.add_argument("--context_length", type=int,   default=256,   help="Block length in tokens (default: 256).")
    argparser.add_argument("--batch_size",     type=int,   default=32,    help="Batch size (default: 32).")
    argparser.add_argument("--lr",             type=float, default=1e-3,  help="Peak learning rate for AdamW (default: 0.001).")
    argparser.add_argument("--warmup_steps",   type=int,   default=200,   help="Linear warm-up steps (default: 200).")
    argparser.add_argument("--hidden_dim",     type=int,   default=256,   help="Model width (default: 256).")
    argparser.add_argument("--n_heads",        type=int,   default=4,     help="Attention heads (default: 4).")
    argparser.add_argument("--ff_dim",         type=int,   default=1024,  help="Feed-forward width (default: 1024).")
    argparser.add_argument("--n_layers",       type=int,   default=2,     help="Transformer layers (default: 2).")
    argparser.add_argument("--dropout",        type=float, default=0.1,   help="Dropout probability (default: 0.1).")
    argparser.add_argument("--seed",           type=int,   default=0,     help="Seed for initialisation and batch order (default: 0).")
    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    print("="*100)
    print(f"[{datetime.now()}] Loading '{args.tok}' tokenizer and building blocks...", flush=True)
    encode, vocab_size = load_tokenizer(args.tok)
    xtr, ytr, _ = make_blocks(encode, "train", args.context_length, seed=args.seed)
    xva, yva, va_chars = make_blocks(encode, "valid", args.context_length)
    tok_per_char = yva.numel() / va_chars
    print(f"    vocab_size={vocab_size}, train blocks={len(xtr)}, valid blocks={len(xva)}, device={DEVICE}")

    model = TransformerLM(vocab_size, args.context_length, args.hidden_dim, args.n_heads,
                          args.ff_dim, args.n_layers, args.dropout).to(DEVICE)
    counts = model.param_counts()
    print(f"    Parameters: {counts}")

    print(f"\n[{datetime.now()}] Training Transformer language model...", flush=True)

    history, meta = train(model, xtr, ytr, xva, yva, DEVICE, args.tok, args.epochs, args.batch_size, args.lr,
          args.warmup_steps, tok_per_char, meta={"args": vars(args), "params": counts})

    save_path = LOG_DIR / "train_run" / f"{args.tok}_train.json"
    save_data({**meta, "history": history}, save_path)
    print(f"\n[{datetime.now()}] Training succesful!!", flush=True)
    print("="*100)


if __name__ == "__main__":
    main()
