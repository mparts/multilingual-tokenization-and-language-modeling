from config import EXTERNAL_CORPUS_DIR
import json


def read_lines(split, lang, path=EXTERNAL_CORPUS_DIR):
    with open(path / split / f"{lang}.txt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_data(data, path, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)