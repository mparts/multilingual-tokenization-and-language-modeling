"""
lanfuage_model.py
"""
from config import MODEL_DIR
from config import LANGS, BOS, EOS
from helpers import read_lines

import math, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm


def make_blocks(encode, split, ctx, langs=LANGS, seed=0):
    sents = [s for l in langs for s in read_lines(split, l)]
    random.Random(seed).shuffle(sents)
    stream = []
    for ids in encode(sents):
        stream += [BOS] + ids + [EOS]
    n = (len(stream) - 1) // ctx
    t = torch.tensor(stream[: n * ctx + 1])
    return t[:-1].view(n, ctx), t[1:].view(n, ctx), sum(len(s) for s in sents)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, hidden_dim=256, n_heads=4, ff_dim=1024, n_layers=2, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, context_length=context_length, hidden_dim=hidden_dim,
                           n_heads=n_heads, ff_dim=ff_dim, n_layers=n_layers, dropout=dropout)
        self.tok, self.pos = nn.Embedding(vocab_size, hidden_dim), nn.Embedding(context_length, hidden_dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        T = x.size(1)
        h = self.drop(self.tok(x) + self.pos(torch.arange(T, device=x.device)))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.head(self.ln(self.blocks(h, mask=mask, is_causal=True)))

    def param_counts(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {"total": n(self), "input_embeddings": n(self.tok),
                "output_layer": n(self.head), "positional": n(self.pos),
                "other": n(self) - n(self.tok) - n(self.head) - n(self.pos)}


def save_checkpoint(path, model, name, meta):
    torch.save({"model_state_dict": model.state_dict(), "config": model.config, "tok": name, "meta": meta}, path)


def load_checkpoint(path, dev):
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    model = TransformerLM(**ckpt["config"]).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt["tok"], ckpt["meta"]


def _autocast(dev):
    return torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")


def bpc_approx(loss, tok_per_char):
    return loss / math.log(2) * tok_per_char


def train(model, xtr, ytr, xva, yva, dev, name, epochs, batch_size, lr, warmup, tok_per_char, meta):
    vocab_size = model.config["vocab_size"]
    steps_per_epoch = len(xtr) // batch_size
    total = epochs * steps_per_epoch

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / total))))

    history, best, step = {"steps": [], "epochs": []}, float("inf"), 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        run, t0 = 0.0, time.time()
        for i in tqdm(range(steps_per_epoch), desc=f"Epoch {ep}/{epochs}", unit="batch", ncols=100,
                      colour="green", disable=not sys.stderr.isatty()):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
            with _autocast(dev):
                logits = model(xb)
            loss = F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run += loss.item()
            if step % 100 == 0:
                history["steps"].append({"step": step, "train_loss": run / (i + 1)})
        vl = evaluate(model, xva, yva, dev)
        rec = {"epoch": ep, "step": step, "train_loss": run / steps_per_epoch,
               "valid_loss": vl, "valid_bpc_approx": bpc_approx(vl, tok_per_char)}
        history["epochs"].append(rec)
        print(rec, f"({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best = vl
            save_checkpoint(MODEL_DIR / f"{name}.pt", model, name, meta)
    return history, meta


def evaluate(model, x, y, dev, bs=64):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb, yb = x[i:i + bs].to(dev), y[i:i + bs].to(dev)
            with _autocast(dev):
                logits = model(xb)
            tot += F.cross_entropy(logits.float().view(-1, logits.size(-1)), yb.view(-1),
                                   reduction="sum").item()
    return tot / y.numel()
