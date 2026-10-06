import shutil
import sys
from datetime import datetime

from config import ROOT, LOG_DIR, VOCAB_DIR, MODEL_DIR, LATEST, MODELS
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
        d = load_json(LOG_DIR / "train_run" / LATEST / f"{name}_train.json")
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
            d = load_json(LOG_DIR / f"{split}_run" / LATEST / f"{name}_{split}_eval.json")
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



if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} <directory-name>")
    dest = ROOT / "data" / "manual_saves" / sys.argv[1]
    dest.mkdir(parents=True, exist_ok=True)
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
