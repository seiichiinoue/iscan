# Infinite SCAN

Implementation of **Infinite SCAN** (Infinite Model of Diachronic Semantic Change) from [Inoue et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.104/).

Infinite SCAN is a nonparametric Bayesian model that tracks how the senses of a target word evolve over time. Given time-stamped context snippets around the target word, it infers sense-specific word distributions and sense probabilities at each time step.

## Overview

| Component | Description |
|-----------|-------------|
| `scan` | Train the model via MCMC (`train.cpp`) |
| `prob` | Print sense probabilities and top context words per time step |
| `eval` | Compute topic coherence (u_mass) of estimated senses |
| `infer` | Compute log-likelihood and perplexity on held-out snippets |
| `tools/create_snippets.py` | Build snippet files from year-split raw corpora (English / Japanese) |
| `tools/create_snippets_using_coha.py` | Build snippets from preprocessed COHA pickle data |
| `tools/create_pseudo_data.py` | Generate synthetic snippet data for experiments |
| `scripts/run.sh` | Example training pipeline on real snippet data |
| `scripts/run_test.sh` | End-to-end pipeline with pseudo data |

### Data format

All training and inference tools expect UTF-8 text with **one snippet per line**:

```
YEAR context_word1 context_word2 ...
```

- `YEAR` is an integer time index (e.g. publication year, or synthetic steps `0`, `1`, …).
- Context words are neighbors of the target word; the **target word itself is not** on the line.
- Each target word is stored in a separate file (e.g. `data/transport.txt`).

## Requirements

**C++ (training / analysis)**

- C++11 compiler, Boost (serialization), GSL, glog, gflags

**Python (data preparation)**

- Python 3, NumPy, NLTK, tqdm, Matplotlib (for `create_pseudo_data.py` plots)

**Locale (required for `scan`, `prob`, `eval`, `infer`)**

The binaries set the C++ locale to `ja_JP.UTF-8` for wide-character I/O. Install and enable it before running:

```sh
sudo apt-get install -y locales
sudo locale-gen ja_JP.UTF-8
export LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8
```

Without this locale, training aborts with `locale::facet::_S_create_c_locale name not valid`.

## Setup

### Option A: Docker (recommended in the original paper workflow)

```sh
docker build -t infinite-scan .
docker run -it -v /path/to/this/repo:/workspace infinite-scan
cd /workspace
export LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8
```

### Option B: Native install (Ubuntu)

```sh
sudo apt-get install -y build-essential libboost-all-dev libgsl-dev \
  libgoogle-glog-dev libgflags-dev locales
sudo locale-gen ja_JP.UTF-8
export LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8

python3 -m pip install numpy matplotlib tqdm nltk
python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet'); nltk.download('stopwords')"

make
```

This builds `scan`, and `make prob`, `make eval`, `make infer` build the analysis tools.

## 1. Prepare data from a time-series corpus

Place **one file per year** under an input directory. For English (COHA-style), each file must have **two header lines** followed by the body text on the third line (see `tests/fixtures/corpora_en/` for a minimal example). Filenames must contain the year after the first underscore, e.g. `coha_1810` (no extra `.txt` in the year token).

```sh
python3 tools/create_snippets.py \
  --lang en \
  --year-start 1810 \
  --year-end 2010 \
  --window-size 5 \
  --input-path /path/to/corpora \
  --output-path ./data \
  transport bank
```

| Option | Description |
|--------|-------------|
| `--lang` | `en` or `ja` |
| `--year-start`, `--year-end` | Year range for snippet keys |
| `--window-size` | Context radius (words on each side of the target) |
| `--input-path` | Directory of per-year corpus files |
| `--output-path` | Writes `<target_word>.txt` per word |
| `--sub-sampling` | Optional subsampling (English) |

**Preprocessed COHA** (pickle from [seiichiinoue/coha](https://github.com/seiichiinoue/coha)):

```sh
python3 tools/create_snippets_using_coha.py \
  --input-path coha.pickle \
  --output-path ./data \
  --year-start 1810 \
  --year-end 2010 \
  --window-size 5 \
  transport
```

## 2. Train the model

```sh
./scan \
  -data_path=./data/pseudo_data.txt \
  -save_path=./results/bin/model.bin \
  -num_iteration=1000 \
  -burn_in_period=500 \
  -min_word_count=10 \
  -start_year=0 \
  -end_year=10 \
  -year_interval=1
```

### Frequently used flags

| Flag | Default | Description |
|------|---------|-------------|
| `-data_path` | `./data/transport/corpus.txt` | Snippet file |
| `-save_path` | `./bin/scan.model` | Output model archive |
| `-num_iteration` | `1000` | MCMC iterations |
| `-burn_in_period` | `500` | Burn-in iterations |
| `-num_sense` | `8` | Maximum number of senses |
| `-start_year`, `-end_year`, `-year_interval` | `1700`, `2020`, `20` | Time bins |
| `-min_word_count` | `-1` | Drop words below this frequency (`-1` = off) |
| `-top_n_word` | `3000` | Vocabulary cap by frequency |
| `-kappa_phi`, `-kappa_psi` | `4.0`, `100.0` | Prior concentration |
| `-from_archive` | `false` | Resume from `-load_path` |

**Wrapper script** (edit paths and years inside the script):

```sh
bash scripts/run.sh transport 3000
```

Expects snippet files under `data/en_v3/<word>.txt` and enough snippets per time bin for stable training.

## 3. Experiments with pseudo data

```sh
mkdir -p results/fig data/pseudo_v3 results/bin results/log

python3 tools/create_pseudo_data.py \
  --num-times 10 \
  --num-senses 2 \
  --context-window-size 10 \
  --vocab-size 10000 \
  --num-sample 10000 \
  --shift-type random \
  --word-prior-type zipf \
  --output-path ./data/pseudo_data.txt

./scan \
  -data_path=./data/pseudo_data.txt \
  -save_path=./results/bin/pseudo.model \
  -min_word_count=10 \
  -start_year=0 \
  -end_year=10 \
  -year_interval=1 \
  -num_iteration=1000
```

| `--shift-type` | Sense trajectories over time |
|----------------|------------------------------|
| `default` | Step change at mid-point |
| `smoothed` | Sigmoid-smoothed change |
| `scurve` | S-curve |
| `random` | Gaussian-process random curves |

**One-shot script** (generates data, trains, writes logs under `results/`):

```sh
bash scripts/run_test.sh 2 3000
```

Arguments: `NUM_SENSES`, `TOP_N_WORD`. The output file must not exist before running `create_pseudo_data.py`.

## 4. Inspect estimated parameters

```sh
./prob -model_path=./results/bin/pseudo.model -use_npmi=true -normalize=true
```

For each time step, prints sense weights and the top 10 context words (NPMI by default). The final `representative:` block lists words aggregated over time per sense.

| Flag | Default | Description |
|------|---------|-------------|
| `-use_npmi` | `true` | Rank words by normalized PMI |
| `-normalize` | `true` | Logistic sense probabilities; `false` shows Gaussian latents |

## 5. Evaluate topic coherence

```sh
./eval -model_path=./results/bin/pseudo.model
```

Prints `coherence: <value>` (u_mass).

## 6. Perplexity on new snippets

```sh
./infer \
  -model_path=./results/bin/pseudo.model \
  -data_path=./data/holdout.txt
```

Uses the **last** training time step for sense assignment. Prints `log_likelihood` and `perplexity`.

## Quick smoke test

After setup, the following sequence was verified on a clean Ubuntu environment:

```sh
export LANG=ja_JP.UTF-8 LC_ALL=ja_JP.UTF-8
make

python3 tools/create_pseudo_data.py \
  --num-times 5 --num-senses 2 --context-window-size 5 \
  --vocab-size 500 --num-sample 500 --shift-type random \
  --word-prior-type zipf --output-path ./data/pseudo_test.txt

./scan -data_path=./data/pseudo_test.txt -save_path=./results/bin/pseudo_test.model \
  -min_word_count=1 -start_year=0 -end_year=5 -year_interval=1 \
  -num_iteration=50 -burn_in_period=10 -num_sense=4 -top_n_word=200

./prob -model_path=./results/bin/pseudo_test.model
./eval -model_path=./results/bin/pseudo_test.model
./infer -model_path=./results/bin/pseudo_test.model -data_path=./data/pseudo_test.txt
```

Snippet extraction can be checked on the tiny fixture under `tests/fixtures/corpora_en/`.

## Project layout

```
train.cpp              # Training entry point → scan
src/scan.hpp           # SCAN model
src/model.hpp          # SCANTrainer (load, train, save)
src/sampler.hpp        # MCMC / Polya-Gamma sampling
src/pg/                # Polya-Gamma auxiliary variable library
tools/                 # prob, eval, infer, data scripts
scripts/               # run.sh, run_test.sh
```

## Citation

```bibtex
@inproceedings{inoue-etal-2022-infinite,
  title     = {Infinite {SCAN}: An Infinite Model of Diachronic Semantic Change},
  author    = {Inoue, Seiichi and Komachi, Mamoru and Ogiso, Toshinobu and Takamura, Hiroya and Mochihashi, Daichi},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages     = {1605--1616},
  year      = {2022},
  url       = {https://aclanthology.org/2022.emnlp-main.104/}
}
```

Seiichi Inoue, Mamoru Komachi, Toshinobu Ogiso, Hiroya Takamura, and Daichi Mochihashi. 2022. [Infinite SCAN: An Infinite Model of Diachronic Semantic Change](https://aclanthology.org/2022.emnlp-main.104/). In *Proceedings of EMNLP 2022*, pages 1605–1616, Abu Dhabi, UAE. Association for Computational Linguistics.
