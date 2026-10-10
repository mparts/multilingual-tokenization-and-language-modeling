# Multilingual Tokenization and Language Modeling
Assignment for the course Machine learning for statistical NLP: Advanced LT2326/LT2926. Department of Philosophy, Linguistics and Theory of Science (FLoV).

## Main repository tree structure
```text
./
├── data/              # contains all data
│   ├── corpus/          # the corpus used
│   ├── manual_saves/    # contains all the saved runs
│   └── vocab/           # the tokenizers
├── docs/              # submission content
├── models/            # holds the trained models
├── src/               # the code
```

## Init
### RUN ALL
To run everything, simply execute this command from the repository root.
```bash
bash RUN.sh
```
The file [run.conf](./run.conf) controls what [RUN.sh](./RUN.sh) executes and how. 

If nothing is altered after a git clone, `RUN.sh`, copies [5_20E_TEST](./data/manual_saves/5_20E_TEST/), and runs evaluation and plotting scripts on the pretrained models.

Keep in mind that this is the only saved_run that has its pretrained models uploaded in the repository.

### Navigating run.conf
You can alter the contents of [run.conf](./run.conf) to change what python scripts are executed. 
```bash
nano run.conf
```
If you decide to replicate any other of the saved_runs or run whatever else you want, keep in mind these important notes:
- Change `RESTORE_NAME` into the name of one of the directories inside [manual_saves](./data/manual_saves/).
- You need to manually alter the model names of `LM_TOKENIZERS`.
- You need to **make sure** that the models in `LM_TOKENIZERS` of [run.conf](./run.conf) and `MODELS` of [config.py](./src/config.py) are **identical**!! Any variation between the two will introduce bugs.

To easily alter `config.py`:
```bash
nano src/config.py
```
### Manual execution
If you do not wish to use `RUN.sh`, you can manually run any of the python scripts inside of [src/](./src/), but keep in mind that some require extra arguments when run.
```python
python3 src/<name_of_module.py> --potential_arguments <arg>
```
Details about the individual scripts can be found in this [README.md](./src/README.md)

## Submission
All files submitted to canvas can also be found inside of [docs](./docs/). My local clone of the repo contains things the online repository doesn't include. Like e.g. all of the trained models, or scattered notes. 

Even though I did execute `RUN.sh` in a fresh clone before submitting in canvas, to make sure that things will work fine for the grader, in the unfortunate case that anything doesn't work, or maybe something more might be needed in the submission, please kindly contact me.