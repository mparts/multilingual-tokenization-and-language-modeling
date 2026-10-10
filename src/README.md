# Implementation Specifics

- Pipeline: **tokenizers → tokenizer analysis → Transformer LM → evaluation → plots**.
- Languages: English, Turkish, Chinese. 
- Every tokenizer is paired with the *same* LM, so differences in results come from the tokenizer only.

Typical run order:
`tokenizer.py` → `examine_tokenizers.py` → `vocab_alocation.py` → `train_lm.py --tok X` → `evaluate_lm.py --tok X` → `plot_*.py` → `save_restore.py`


## TOKENIZER

### tokenizer.py
Three tokenizer families, named `<Family>_<vocab size>` and selected via `MODELS` in `config.py`.

| Family | Pre-tokenizer | Notes |
|---|---|---|
| **Character-Level** | none | Vocab = specials + every training character, sorted by frequency. Unseen chars map to `<UNK>`. ~9k entries (mostly Chinese). |
| **ByteLevelBPE** | `ByteLevel` | Starts from all 256 bytes, so nothing is ever `<UNK>`. Sizes 2k–20k. Chosen as the most sensible option for multilingual text. |
| **CharLevelBPE** | `Metaspace` (`▁`) | Starts from characters. `limit_alphabet = vocab - specials` keeps every character, so vocab must exceed the ~9k char alphabet to be meaningful (hence 10k+). Smaller sizes would drop characters instead of merging. |

- Special tokens have fixed ids across all tokenizers: `<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3`.
- Stats are computed on the **validation** split: total tokens, tokens/sentence, chars/token, `<UNK>` %.

### examine_tokenizers.py
Prints and saves `N_SAMPLES` random validation sentences per language, split into pieces by each tokenizer. Purely for manual inspection.

### vocab_alocation.py
Answers: *which language "owns" each vocabulary entry?* (BPE tokenizers only.)
- Counts are **per character, not per token**, since the three corpora have similar character counts but very different token counts.
- Token class = language if its share of the usage is ≥ `THRESH` (0.9), else `shared`. Tokens used < `MIN_COUNT` (5) times are `rare`.
- Also reports: share of each language's token stream per class, byte-fragment tokens (decode to `�`), and Jaccard overlap of used tokens between language pairs.

### plot_vocab.py
Vocabulary size vs. tokens/sentence, chars/token and total tokens, per language. ByteLevel and CharLevel families are on separate panels.

## TRANSFORMER

### language_model.py
- **Model:** small decoder-only Transformer (`TransformerEncoder` + causal mask). Pre-norm, GELU, learned positional embeddings. Defaults: 2 layers, hidden 256, 4 heads, FF 1024, dropout 0.1. Input embedding and output layer are **not tied**, and their parameters are counted separately, since vocab size mostly changes those two.
- **Data (`make_blocks`):** sentences from all languages are shuffled (fixed seed), joined as `<BOS> ids <EOS>` into one stream, and cut into fixed-length blocks (default 256). No padding is needed, so `<PAD>` is never used. Fixed seed means every tokenizer sees the same sentence order.
- **Training:** AdamW, lr 1e-3, linear warm-up then cosine decay, gradient clipping 1.0, bf16 autocast on GPU (loss computed in fp32). The incomplete last batch is dropped. The checkpoint with the best validation loss is kept.
- **Metric:** loss per token is **not comparable** across tokenizers (different tokens per text), so we report approximate **bits per character**: `loss / ln 2 × tokens/char`. It is approximate because `<BOS>`/`<EOS>` count as tokens but not as characters.

### Design choices and what they affect
Each item below can be changed or removed to get different results.

**Architecture**
- **Causal mask (`is_causal=True`)**: required. Without it the model sees the next token and the loss collapses to a meaningless low value.
- **Pre-norm (`norm_first=True`)**: LayerNorm before each sub-layer trains more stably at high learning rates than post-norm and needs less warm-up.
- **GELU**: smoother than ReLU, which is the usual choice for Transformer LMs. The effect on results is small.
- **Learned positional embeddings**: simple, but the model cannot handle sequences longer than `context_length`. Sinusoidal or rotary encodings would remove that limit.
- **Untied embeddings and output layer**: gives the model more parameters, and these grow with vocab size. Tying them would shrink the large-vocab models (e.g. 20k) and make the parameter budget fairer across tokenizers.
- **Dropout 0.1**: a regulariser. With a small model and a few epochs it matters little, but overfitting becomes visible if you train longer or use a smaller corpus.
- **Model size (2 layers, 256 hidden)**: kept small and identical for all tokenizers so training is fast and tokenizer effects are not hidden by model differences. Larger models would likely lower BPC across the board.

**Optimisation**
- **AdamW**: Adam with decoupled weight decay (PyTorch default 0.01, not tuned).
- **Warm-up + cosine decay (`LambdaLR`)**: the lr rises linearly for `warmup` steps, then follows a cosine down to 0 at the last step. Warm-up avoids early divergence. Without decay, the final loss is usually a bit worse and noisier. `warmup` and `lr` interact: a higher peak lr needs a longer warm-up.
- **Gradient clipping (norm 1.0)**: caps the size of each update. This guards against loss spikes at lr 1e-3, which is fairly high. Skipping it works, but rare bad batches can destabilise training.
- **bf16 autocast (GPU only)**: runs the forward pass in bfloat16 for speed and lower memory. The loss is computed in fp32 for numerical accuracy. bf16 has fp32's range, so no `GradScaler` is needed.
- **Per-epoch reshuffling of blocks (`randperm`)**: gives a different batch order each epoch. The last incomplete batch is dropped so every step has the same batch size.
- **Best-checkpoint saving**: the saved model is the epoch with the lowest validation loss, not the last one. This protects against overfitting in later epochs.

**Data**
- **Concatenated stream cut into blocks**: no padding, so no wasted compute and `<PAD>` is never used. The downside is that blocks start mid-sentence, and `<BOS>`/`<EOS>` are the only boundary markers.
- **`context_length` (default 256)**: this is in *tokens*, so it covers different amounts of text per tokenizer. e.g. Character-level sees far fewer characters per block than a 20k BPE.
- **Fixed shuffle seed for valid/test**: all tokenizers are evaluated on the same sentence order.

**Metric**
- **Loss vs. perplexity vs. BPC**: loss and perplexity are per token and only comparable within one tokenizer. BPC is normalised per character, so it is the number to use when comparing tokenizers. It is approximate because of the `<BOS>`/`<EOS>` tokens (counted as tokens, not as characters).

### train_lm.py / evaluate_lm.py
CLI wrappers (`--tok` selects tokenizer, hyperparameters are arguments, defaults above). Training saves args, parameter counts and per-epoch history to JSON. Evaluation reports loss, perplexity and BPC on `valid` or `test`, for all languages combined and for each language separately (blocks rebuilt per language).

### plot_lm.py
Learning curves (train/valid loss, valid BPC) and per-language BPC bar chart and heatmap.

## UTILITIES

### config.py
Paths, language list, special-token ids, device, and `MODELS` (the active tokenizer set, keep one line uncommented). `EXTERNAL_CORPUS_DIR` points to the shared cluster data, switch to the internal one when running elsewhere.

### helpers.py
Corpus reading, JSON load/save, tokenizer loading. `load_tokenizer` gives every tokenizer the same interface: `encode(list of str) -> list of id lists`, plus vocab size.

### save_restore.py
`--mode save` merges per-model train/eval JSONs into a single `train.json` / `valid.json` / `test.json` (shared args stored once, per-model differences flagged) and copies models, vocabs, plots and logs into `data/manual_saves/<name>`. `--mode restore` copies them back.