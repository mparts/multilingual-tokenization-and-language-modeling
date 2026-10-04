#!/usr/bin/env bash
#
# Script: RUN.sh
# Description: Runs the whole pipeline. Logs all output.
# Usage: bash RUN.sh
#

cd "$(dirname "$0")" || exit 1 
source "run.conf"

LINE="===================================================================================================="

if $APPEND_LOG; then
    echo "$LINE" | tee -a "$LOG_FILE"
else
    echo "$LINE" | tee "$LOG_FILE"
fi

printf "    |    |\n    |    |\n    |    |\nMultilingual Tokenization and Language Modeling\n    |    |\n    |    |\n    |    |\n" | tee -a "$LOG_FILE"
echo "$LINE" | tee -a "$LOG_FILE"

section() {
    printf "    |    |\n    |    |\n%s\n    |    |\n    |    |\n" "$1" | tee -a "$LOG_FILE"
}

# run_step <true|false> <script> <args...>: runs src/<script> (logged) or reports that it was skipped
run_step() {
    local enabled=$1 script=$2
    shift 2
    if $enabled; then
        echo "~Running $script script..." | tee -a "$LOG_FILE"
        $PYTHON "src/$script" "$@" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "<!>    $script skipped.    <!>" | tee -a "$LOG_FILE"
    fi
}

# Tokenizers
section "Part 1. Tokenizers"
run_step $TRAIN_TOKENIZERS tokenizer.py
run_step $EXAMINE_TOKENIZERS examine_tokenizers.py

# Language modeling
section "Part 2. Language modeling"
for TOK in $LM_TOKENIZERS; do
    run_step $TRAIN_LM train_lm.py --tok $TOK --epochs $EPOCHS_LM --context_length $CONTEXT_LENGTH --batch_size $BATCH_SIZE --lr $LEARNING_RATE --hidden_dim $HIDDEN_DIM --n_heads $N_HEADS --ff_dim $FF_DIM --n_layers $N_LAYERS --dropout $DROPOUT --seed $SEED
    run_step $EVAL_LM evaluate_lm.py --tok $TOK --split $EVAL_SPLIT

    sed -i '/^[[:space:]]*Epoch/d' "$LOG_FILE"
done

# Plotting
section "Part 3. Plotting"
run_step $PLOT_LM plot_lm.py --split $EVAL_SPLIT

# Saving
section "Part 4. Saving"
run_step $SAVE_RUN save_latest.py $RUN_NAME

echo "^^^                                       ^^^" | tee -a "$LOG_FILE"
echo "^^^    End of RUN.sh pipeline.            ^^^" | tee -a "$LOG_FILE"
echo "^^^                                       ^^^" | tee -a "$LOG_FILE"

exit 0
