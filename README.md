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
$ python3 scripts/create_snippets.py --lang LANG --year_start YEAR_START --year_end YEAR_END --window_size WINDOW_SIZE --input_path INPUT_PATH --output_path OUTPUT_PATH TARGET_WORD1 TARGET_WORD2 ...
```

## Run

compile and train.

```
$ make
$ ./scan -num_iteration=1000 -top_n_word=1000 -data_path=PATH_TO_DATA -save_path=PATH_TO_MODEL
```

## Estimation with pseudo data

run to generate pseudo data and train model.

```
$ make prob
$ sh tests/test.sh NUM_SENSES
```

## References

- [A Bayesian Model of Diachronic Meaning Change. (2016). L. Frermann and M. Lapata.](https://www.aclweb.org/anthology/Q16-1003.pdf)
- [Logistic Stick-Breaking Process. (2011). L. Ren et al.](https://www.jmlr.org/papers/volume12/ren11a/ren11a.pdf)
- [Dependent Multinomial Models Made Easy: Stick Breaking with the Polya-Gamma Augmentation. (2015). S. W. Linderman et al.](https://www.cs.princeton.edu/~rpa/pubs/linderman2015multinomial.pdf)