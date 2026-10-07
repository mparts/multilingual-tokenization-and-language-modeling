import shutil
import sys
from datetime import datetime
from argparse import ArgumentParser

from config import ROOT, LOG_DIR, VOCAB_DIR, MODEL_DIR, MODELS
from helpers import load_json, save_data

EPOCH_KEYS = ["epoch", "step", "train_loss", "valid_loss", "valid_bpc_approx", "train_time"]


def copy(src, dst):
    if not src.exists():
        print(f"  [!] missing: {src.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def split_args(args_per_model):
    base = next(iter(args_per_model.values()))
    shared = {k: v for k, v in base.items() if k != "tok"}
    overrides = {}
    for name, args in args_per_model.items():
        diff = {k: v for k, v in args.items() if k != "tok" and shared.get(k) != v}
        if diff:
            overrides[name] = diff
            print(f"  [!] {name} used different args than the others: {diff}")
    return shared, overrides


def merge_train(dest):
    args, runs = {}, {}
    for name in MODELS:
        d = load_json(LOG_DIR / "train_run" / f"{name}_train.json")
        if d is None:
            continue
        args[name] = d["args"]
        epochs = [{k: e[k] for k in EPOCH_KEYS} for e in d["history"]["epochs"]]
        runs[name] = {"params": d["params"], "epochs": epochs}
    if not runs:
        return
    shared, overrides = split_args(args)
    out = {"saved_at": datetime.now().isoformat(timespec="seconds"), "models": list(runs),
           "shared_args": shared, **({"args_overrides": overrides} if overrides else {}), "runs": runs}
    save_data(out, dest / "train.json")


def merge_eval(dest):
    args, results, splits = {}, {}, set()
    for name in MODELS:
        for split in ("valid", "test"):
            d = load_json(LOG_DIR / f"{split}_run" / f"{name}_{split}_eval.json")
            if d is not None:
                splits.add(split)
                args[(split, name)] = d["args"]
                results.setdefault(split, {})[name] = d["results"]
    if not results:
        print("  [!] no evaluation files found")
        return
    for split, res in results.items():
        shared, overrides = split_args({n: args[(split, n)] for n in res})
        missing = [n for n in MODELS if n not in res]
        if missing:
            print(f"  [!] {split}: no eval for {missing}")
        out = {"saved_at": datetime.now().isoformat(timespec="seconds"), "split": split, "models": list(res),
               "shared_args": shared, **({"args_overrides": overrides} if overrides else {}), "results": res}
        save_data(out, dest / f"{split}.json")


def save(dest):
    print(f"Saving {len(MODELS)} models to {dest.relative_to(ROOT)}/")

    merge_train(dest)
    merge_eval(dest)

    copy(ROOT / "run.log", dest / "run.log")
    for name in MODELS:
        copy(MODEL_DIR / f"{name}.pt", dest / "models" / f"{name}.pt")
        copy(VOCAB_DIR / f"{name}_vocab.json", dest / "vocab" / f"{name}_vocab.json")
    copy(LOG_DIR / "plots", dest / "plots")
    copy(LOG_DIR / "tokenizers", dest / "tokenizers")
    print(f"\n[{datetime.now()}] Done!!", flush=True)


def restore(dest):
    if not dest.exists():
        sys.exit(f"[!] Checkpoint does not exist: {dest.relative_to(ROOT)}")

    print(f"Restoring from {dest.relative_to(ROOT)}/")
    copy(dest / "models", MODEL_DIR)
    copy(dest / "plots", LOG_DIR / "plots",)
    copy(dest / "tokenizers", LOG_DIR / "tokenizers",)
    copy(dest / "vocab", VOCAB_DIR)
    print(f"\n[{datetime.now()}] Restore done!!", flush=True)


if __name__ == "__main__":
    argparser = ArgumentParser(description="Saves a 'checkpoint', or restores it to or from the specified path.")
    argparser.add_argument("--save_dir", type=str,  required=True, help="Path to save to or reload from.")
    argparser.add_argument("--mode",     type=str,  default="save", choices=["save", "restore"], help="save or restore?")
    args = argparser.parse_args()
    
    save_dir = ROOT / "data" / "manual_saves" / args.save_dir
    if args.mode == "save":
        save_dir.mkdir(parents=True, exist_ok=True)
        save(save_dir)
    elif args.mode == "restore":
        restore(save_dir)
