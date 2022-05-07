# scripts for dataset building
# target corpus:
# english: COHA, japanese: CHJ
# TODO: support both japaense word segmentation and pos-tagging

import os, sys
import pickle
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger

np.random.seed(0)

# global objects
lemmatizer = WordNetLemmatizer()
tagger = PerceptronTagger()
word_freq = defaultdict(int)
sum_word_freq = None
pos2id = {'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'JJ': 'a', 'JJR': 'a', 'JJS': 'a'}
available_pos_en = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
available_pos_ja = ['名詞', '形容詞', '動詞', '形状詞', '副詞']
stop_words = set(stopwords.words('english'))
stop_words |= {"'", '"', ':', ';', '.', ',', '-', '--', '...', '//', '/', '!', '?', "'s", "@", "<p>", "(", ")", "・"}

def _preprocess_english(sents):
    sents = tagger.tag_sents(sents)
    ret = []
    for line in sents:
        line = [(w, pos) for w, pos in line if pos in available_pos_en]
        line = [lemmatizer.lemmatize(w.lower(), pos2id[pos]) for w, pos in line if not w.lower() in stop_words and w.isalpha()]
        ret.append(line)
    return ret

def _precalc_statistics(corpora):
    global word_freq, sum_word_freq
    for year, corpus in corpora:
        for line in corpus:
            for word_pos in line:
                if len(word_pos) < 4 or "_" not in word_pos:
                    continue
                word, pos = word_pos.split("_")
                word_freq[word] += 1
    sum_word_freq = sum(word_freq.values())
    return None

def _preprocess_japanese(sents):
    def _remove_prob(x):
        return 1.0 - np.sqrt(1e-4 / float(word_freq[x] / sum_word_freq))
    ret = []
    for line in sents:
        line = [word_pos.split("_") for word_pos in line if len(word_pos) > 3 and "_" in word_pos]
        line = [w for (w, p) in line 
                if p.find("記号") == -1
                and p.find("数") == -1
                and p in available_pos_ja
                and w not in stop_words
                and (1 if _remove_prob(w) < 0 else 1 - np.random.binomial(1, _remove_prob(w)))]
        ret.append(line)
    return ret

def _remove_unnecessary_sents(corpus, target_words):
    def check(sentence):
        for tar in target_words:
            if tar in sentence:
                return True
        return False
    return [sent for sent in corpus if check(sent)]

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
                with open(os.path.join(output_path, tar_word+".txt"), "a") as f:
                    f.write(f"{str(year)} {' '.join(snippet)}\n")

def load_japanese_corpora(corpora_path):
    files = os.listdir(corpora_path)
    corpora = []
    for fn in tqdm(files):
        corpus = [sent.strip().split() for sent in open(os.path.join(corpora_path, fn)).readlines()]
        year = int(fn.split(".")[0])
        corpora.append([year, corpus])
    return corpora

def load_english_corpora(corpora_path):
    files = os.listdir(corpora_path)
    corpora = []
    for fn in tqdm(files):
        with open(os.path.join(corpora_path, fn)) as f:
            f.readline()  # header
            f.readline()  # header
            sentences = f.readline().replace(' ? ', '\n').replace(' . ', '\n').replace(' ! ', '\n').split('\n')
            sentences = [sent.strip().split() for sent in sentences]
            year = int(fn.split("_")[1])
            corpora.append([year, sentences])
    return corpora

def load_corpora(corpora_path, lang):
    assert lang in ["en", "ja"], f'language {lang} is not supported'
    if lang == "en":
        return load_english_corpora(corpora_path)
    elif lang == "ja":
        return load_japanese_corpora(corpora_path)

parser = argparse.ArgumentParser()
parser.add_argument('target_words', type=str, nargs='+')
parser.add_argument('--lang', default='en', type=str)
parser.add_argument('--year-start', default=1800, type=int)
parser.add_argument('--year-end', default=2010, type=int)
parser.add_argument('--window-size', default=5, type=int)
parser.add_argument('--input-path', default='coha', type=str)
parser.add_argument('--output-path', default='data', type=str)
args = parser.parse_args()

corpora = load_corpora(args.input_path, args.lang)
print(f"target words: {str(args.target_words)}")
create_snippets(corpora=corpora,
                target_words=args.target_words,
                lang=args.lang,
                year_start=args.year_start,
                year_end=args.year_end,
                window_size=args.window_size,
                output_path=args.output_path)
