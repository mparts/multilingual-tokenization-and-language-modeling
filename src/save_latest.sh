#!/usr/bin/env bash
#
# Script: save_latest.sh
# Description: Saves the latest run to a specified directory.
# Usage: bash save_latest.sh <directory-name>
#

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$1" ]]; then
    echo "Usage: $0 <directory-name>"
    exit 1
fi

DEST_DIR="$SCRIPT_DIR/../data/manual_saves/$1"

mkdir -p "$DEST_DIR"

cp "$SCRIPT_DIR/../run.log" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../models" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../data/vocab" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../data/logs/plots" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../data/logs/tokenizers" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../data/logs/train_run/latest_runs" "$DEST_DIR/"
cp -r "$SCRIPT_DIR/../data/logs/valid_run/latest_runs" "$DEST_DIR/"