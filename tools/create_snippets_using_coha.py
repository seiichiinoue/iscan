"""
script for creating snnipets
before run this script, need to create processed data using seiichiinoue/coha
"""
import os, sys
import pickle
import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords 


np.random.seed(0)

# global objects
word_freq = defaultdict(int)
sum_word_freq = None
available_pos_en = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
available_pos_ja = ['名詞', '形容詞', '動詞', '形状詞', '副詞']
stop_words = set(stopwords.words('english'))
stop_words |= {"'", '"', ':', ';', '.', ',', '-', '--', '...', '//', '/', '!', '?', "'s", "@", "<p>", "(", ")", "・"}

def _precalc_english(data):
    global word_freq, sum_word_freq
    for year, line in data:
        for word_lemma_pos in line:
            word, lemma, pos = word_lemma_pos
            word_freq[word] += 1
    sum_word_freq = sum(word_freq.values())
    return None

def _preprocess_english(data, sub_sampling=False):
    def _remove_prob(x):
        return 1.0 - np.sqrt(1e-3 / float(word_freq[x] / sum_word_freq))
    ret = []
    for year, line in data:
        line = [lemma for word, lemma, pos in line if pos in available_pos_en]
        if sub_sampling:
            line = [lemma.lower() for lemma in line 
                    if not lemma.lower() in stop_words 
                    and lemma.isalpha()
                    and (1 if _remove_prob(lemma) < 0 else 1 - np.random.binomial(1, _remove_prob(lemma)))]
        else:
            line = [lemma.lower() for lemma in line 
                    if not lemma.lower() in stop_words 
                    and lemma.isalpha()]
        ret.append([int(year), line])
    return ret

def create_snippets(data,
                    target_words,
                    year_start=1800,
                    year_end=2010,
                    window_size=5,
                    output_path="data",
                    sub_sampling=False):
    snippets = {i: {target_words[j]: [] for j in range(len(target_words))} for i in range(year_start, year_end+1)}
    _precalc_english(data)
    data = _preprocess_english(data)
    for year, line in tqdm(data):
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

def load_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    return data

parser = argparse.ArgumentParser()
parser.add_argument('target_words', type=str, nargs='+')
parser.add_argument('--year-start', default=1810, type=int)
parser.add_argument('--year-end', default=2010, type=int)
parser.add_argument('--window-size', default=5, type=int)
parser.add_argument('--input-path', default='coha.pickle', type=str)
parser.add_argument('--output-path', default='data', type=str)
parser.add_argument('--sub-sampling', action='store_true')
args = parser.parse_args()

data = load_data(args.input_path)
print(f"target words: {str(args.target_words)}")
create_snippets(data=data,
                target_words=args.target_words,
                year_start=args.year_start,
                year_end=args.year_end,
                window_size=args.window_size,
                output_path=args.output_path,
                sub_sampling=args.sub_sampling)
