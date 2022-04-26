# Infinite Mixture Model of Diachronic Meaning Change

This is an implementation of the Infinite SCAN in "Infinite SCAN: Joint Estimation of Changes and the Number of Word Senses with Gaussian Markov Random Fields."

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

run to generate pseudo data and estimate model.

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

## References

- [A Bayesian Model of Diachronic Meaning Change. (2016). L. Frermann and M. Lapata.](https://www.aclweb.org/anthology/Q16-1003.pdf)
- [Logistic Stick-Breaking Process. (2011). L. Ren et al.](https://www.jmlr.org/papers/volume12/ren11a/ren11a.pdf)
- [Dependent Multinomial Models Made Easy: Stick Breaking with the Polya-Gamma Augmentation. (2015). S. W. Linderman et al.](https://www.cs.princeton.edu/~rpa/pubs/linderman2015multinomial.pdf)