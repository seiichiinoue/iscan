# Infinite Model of Diachronic Semantic Change

This is an implementation of infinite model of diachronic semantic change: Infinite SCAN.

## Environment

using docker to run code in this repository.

```
$ docker build -t boost .
$ docker run -it -v LOCAL_PATH:CONTAINER_PATH boost
```

## Create snippets

before run command, you need to prepare time-series corpora.

corpora should be placed in an arbitrary directory consisting of a group of files by year.

```
$ python3 tools/create_snippets.py --lang LANG --year-start YEAR_START --year-end YEAR_END --window-size WINDOW_SIZE --input-path INPUT_PATH --output-path OUTPUT_PATH TARGET_WORD1 TARGET_WORD2 ...
```

## Estimation

compile and estimate model.

```
$ make
$ ./scan -num_iteration=1000 -min_word_count=10 -data_path=PATH_TO_DATA -save_path=PATH_TO_MODEL
```

## Estimation with pseudo data

generate pseudo data and estimate model.

```
$ python3 tools/create_snippets.py --num-times NUM_TIMES --num-senses NUM_SENSES --context-window-size WINDOW_SIZE --vocab-size-per-sense VOCAB_SIZE --num-sample NUM_SAMPLE --shift-type random --word-prior-type zipf --output-path PATH_TO_DATA
$ ./scan -data_path=PATH_TO_DATA -save_path=PATH_TO_SAVE -min_word_count=MIN_WORD_COUNT -start_year=0 -end_year=NUM_TIMES -year_interval=1 -num_iteration=NUM_ITERATION
```

## Output estimated parameters

build and run.

```
$ make prob
$ ./prob -model_path=PATH_TO_MODEL -use_npmi=true
```

## Reference

Seiichi Inoue, Mamoru Komachi, Toshinobu Ogiso, Hiroya Takamura, and Daichi Mochihashi. 2022. Infinite SCAN: [An Infinite Model of Diachronic Semantic Change](https://aclanthology.org/2022.emnlp-main.104/). In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 1605–1616, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.