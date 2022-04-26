# scripts for dataset building
# TODO: support both japaense word segmentation and pos-tagging

import os, sys
import pickle
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

import MeCab
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger


# global objects
lemmatizer = WordNetLemmatizer()
tagger = PerceptronTagger()
word_freq = defaultdict(int)
sum_word_freq = None
pos2id = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a'}
available_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
stop_words = set(stopwords.words('english'))
stop_words |= {"'", '"', ':', ';', '.', ',', '-', '--', '...', '//', '/', '!', '?', "'s", "@", "<p>", "(", ")"}

def _remove_unnecessary_sents(corpus, target_words):
    def check(sentence):
        for tar in target_words:
            if tar in sentence:
                return True
        return False
    return [sent for sent in corpus if check(sent)]

def _preprocess_english(sents):
    sents = tagger.tag_sents(sents)
    ret = []
    for line in sents:
        line = [(w, pos) for w, pos in line if pos in available_pos]
        line = [lemmatizer.lemmatize(w.lower(), pos2id[pos]) for w, pos in line if not w.lower() in stop_words and w.isalpha()]
        ret.append(line)
    return ret

def _precalc_statistics(corpora):
    global word_freq, sum_word_freq
    for year, corpus in corpora:
        for line in corpus:
            for word in line:
                word_freq[word] += 1
    sum_word_freq = sum(word_freq.values())
    return None

def _preprocess_japanese(sents):
    def _remove_prob(x):
        return 1.0 - np.sqrt(1e-4 / float(word_freq[x] / sum_word_freq))
    ret = []
    for line in sents:
        line = [w for w in line 
                if not w.isnumeric() 
                and w not in stop_words
                and (1 if _remove_prob(w) < 0 else 1 - np.random.binomial(1, _remove_prob(w)))]
        ret.append(line)
    return ret

def create_snippets(corpora,
                    target_words,
                    lang="en",
                    year_start=1800,
                    year_end=2010,
                    window_size=5,
                    output_path="data"):
    snippets = {i: {target_words[j]: [] for j in range(len(target_words))} for i in range(year_start, year_end+1)}
    if lang == "ja":
        _precalc_statistics(corpora)
    for year, corpus in tqdm(corpora):
        if lang == "en":
            corpus = _remove_unnecessary_sents(corpus, target_words)
            corpus = _preprocess_english(corpus)
        elif lang == "ja":
            corpus = _preprocess_japanese(corpus)
        for line in corpus:
            if len(line) < 2:
                continue
            for i, word in enumerate(line):
                if word in target_words:
                    lb, rb = max(0, i - window_size), min(len(line), i + window_size + 1)
                    snippets[year][word].append(line[lb:i]+line[i+1:rb])
    for year, snippets_y in snippets.items():
        for tar_word, snippets_y_w in snippets_y.items():
            for snippet in snippets_y_w:
                if not os.path.exists(os.path.join(output_path, tar_word)):
                    os.mkdir(os.path.join(output_path, tar_word))
                with open(os.path.join(output_path, tar_word, "snippets.txt"), "a") as f:
                    f.write(f"{str(year)} {' '.join(snippet)}\n")

def load_corpora(corpora_path):
    files = os.listdir(corpora_path)
    corpora = []
    for fn in files:
        corpus = [sent.strip().split() for sent in open(os.path.join(corpora_path, fn)).readlines()]
        year = int(fn.split(".")[0])
        corpora.append([year, corpus])
    return corpora

parser = argparse.ArgumentParser()
parser.add_argument('target-words', type=str, nargs='+')
parser.add_argument('--lang', default='en', type=str)
parser.add_argument('--year-start', default=1800, type=int)
parser.add_argument('--year-end', default=2010, type=int)
parser.add_argument('--window-size', default=5, type=int)
parser.add_argument('--input-path', default='coha', type=str)
parser.add_argument('--output-path', default='data', type=str)
args = parser.parse_args()

corpora = load_corpora(args.input_path)
print(f"target words: {str(args.target_words)}")
create_snippets(corpora=corpora,
                target_words=args.target_words,
                lang=args.lang,
                year_start=args.year_start,
                year_end=args.year_end,
                window_size=args.window_size,
                output_path=args.output_path)
