from os import path
import numpy as np
import argparse

PROJECT_TOP = "/workspace"
OUTPUT_PREFIX = path.join(PROJECT_TOP, "tests/sampled")
OUTPUT_PATH = path.join(OUTPUT_PREFIX, "pseudo_data.txt")

class Sense:
    def __init__(self):
        self.priors = []

    def set_probs(self, t: int = 16, change_point: int = 8, reverse: bool = False):
        assert t > change_point
        self.priors = [10 for _ in range(change_point)]
        if reverse:
            self.priors = self.priors + [0 for _ in range(t - change_point)]
        else:
            self.priors = [0 for _ in range(t - change_point + 1)] + self.priors[:-1]

    def set_smoothed_probs(self, t: int = 16, change_point: int = 8, reverse: bool = False):
        def f(x):
            return 1 / (1 + np.exp(-x))
        
        assert t > change_point
        slices = np.arange(-3, 3, 6/t)
        if reverse:
            slices = list(reversed(slices))
        for i in range(t):
            self.priors.append(f(slices[i]))

    def set_scurve_probs(self, t: int = 16, change_point: int = 8, reverse: bool = False):
        assert t > change_point
        slices = np.arange(-3, 3, 6/t)
        for i in range(t):
            if reverse:
                self.priors.append(-1 * slices[i])
            else:
                self.priors.append(slices[i])

class Sampler:
    def __init__(self,
                 num_times: int = 16,
                 num_senses: int = 2,
                 context_window_size: int = 10,
                 vocab_size_per_sense: int = 3,
                 ratio_common_vocab: float = 0.2,
                 shift_type: str = "default"):
        self.num_times = num_times
        self.num_senses = num_senses
        self.context_window_size = context_window_size
        self.senses = []
        for i in range(self.num_senses):
            sense = Sense()
            if shift_type == "default":
                sense.set_probs(self.num_times, reverse=bool(i))
            elif shift_type == "s-curve":
                assert num_senses == 2
                sense.set_scurve_probs(self.num_times, reverse=bool(i))
            elif shift_type == "smoothed":
                sense.set_smoothed_probs(self.num_times, reverse=bool(i))
            self.senses.append(sense)
        xs = np.array([s.priors for s in self.senses]).T
        self.probs = np.array([self.softmax(x) for x in xs])
        self.vocab_size_per_sense = vocab_size_per_sense
        self.ratio_common_vocab = ratio_common_vocab
        # hypothesize uniform distribution
        # vocab size = num_sense * vocab_size_per_sense
        # e.g. num_sense = 2, vocab_size_per_sense = 3 then vocab size is 6
        self.word_probs = []
        for sense in range(self.num_senses):
            start_idx = sense * self.vocab_size_per_sense
            end_idx = (sense + 1) * self.vocab_size_per_sense
            probs = [0 if i < start_idx or i >= end_idx 
                     else (1.0 / self.vocab_size_per_sense)
                     for i in range(self.vocab_size_per_sense * self.num_senses)]
            self.word_probs.append(probs)
        self.id_to_token = []
        num_common_vocab = int(self.vocab_size_per_sense * self.ratio_common_vocab)
        for k in range(self.num_senses):
            for i in range(self.vocab_size_per_sense - num_common_vocab):
                self.id_to_token.append("sense{}_word{}".format(str(k), str(i)))
            for i in range(num_common_vocab):
                self.id_to_token.append("common_word{}".format(str(i)))

    def softmax(self, x):
        ret = np.exp(x) / np.sum(np.exp(x))
        return ret

    def draw_sense(self, t):
        assert t < self.num_times
        return np.random.multinomial(1, self.probs[t])

    def draw_words(self, n_sample: int = 100, imbalance: bool = False):
        assert path.exists(OUTPUT_PREFIX)
        with open(OUTPUT_PATH, "w") as f:
            for t in range(self.num_times):
                # reduce the number of old sample
                if t < 5 and imbalance:
                    _n_sample = int(n_sample * 0.1)
                else:
                    _n_sample = n_sample
                for n in range(_n_sample):
                    sense = self.draw_sense(t).argmax()
                    words = np.random.multinomial(
                        self.context_window_size, self.word_probs[sense])
                    snippet = []
                    for word_id, n_word in enumerate(words):
                        snippet.extend([self.id_to_token[word_id] for _ in range(n_word)])
                    f.write("{} {}\n".format(str(t), " ".join(snippet)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-times', type=int, default=16)
    parser.add_argument('--num-senses', type=int, default=2)
    parser.add_argument('--context-window-size', type=int, default=10)
    parser.add_argument('--vocab-size-per-sense', type=int, default=10)
    parser.add_argument('--ratio-common-vocab', type=float, default=0.2)
    parser.add_argument('--num-sample', type=int, default=100)
    parser.add_argument('--shift-type', type=str, default="s-curve")
    parser.add_argument('--output-path', type=str, default="pseudo_data.txt")
    args = parser.parse_args()

    OUTPUT_PATH = args.output_path

    sampler = Sampler(
        num_times=args.num_times,
        num_senses=args.num_senses,
        context_window_size=args.context_window_size,
        vocab_size_per_sense=args.vocab_size_per_sense,
        ratio_common_vocab=args.ratio_common_vocab,
        shift_type=args.shift_type
        )
    sampler.draw_words(n_sample=args.num_sample)