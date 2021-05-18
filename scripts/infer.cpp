#include "src/model.hpp"

void load_documents(string filepath, SCANTrainer &trainer, vector<vector<size_t>> &dataset, unordered_map<size_t, int> &word_frequency) {
    wifstream ifs(filepath.c_str());
    assert(ifs.fail() == false);
    wstring sentence;
    while (getline(ifs, sentence) && !ifs.eof()) {
        int doc_id = dataset.size();
        dataset.push_back(vector<size_t>());
        vector<wstring> words;
        split_word_by(sentence, L' ', words);
        if (words.size() == 0) continue;
        for (auto word : words) {
            if (word.size() == 0) continue;
            size_t word_id;
            if (!(trainer._vocab->word_exists(word))) {
                // unknown word id
                word_id = trainer._vocab->num_words();
            } else {
                word_id = trainer._vocab->get_word_id(word);
            }
            dataset[doc_id].push_back(word_id);
            word_frequency[word_id] += 1;
        }
    }
}
double compute_log_likelihood(SCANTrainer &trainer, vector<vector<size_t>> &dataset) {
    trainer._update_logistic_Phi();
    trainer._update_logistic_Psi();
    double log_pw = 0.0;
    for (int n=0; n<dataset.size(); ++n) {
        // calculation for $\phi^t_k$
        double* probs_n = trainer._probs;
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            probs_n[k] = log(trainer._logistic_Phi[trainer._scan->_n_t-1][k]);
        }
        // calculation for $\prod \psi^{t, k}_wi$
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            for (int i=0; i<trainer._scan->_context_window_width; ++i) {
                size_t word_id = dataset[n][i];
                if (word_id == trainer._vocab->num_words()) continue;
                if (trainer._word_frequency[word_id] < trainer._ignore_word_count) {
                    continue;
                }
                probs_n[k] += log(trainer._logistic_Psi[trainer._scan->_n_t-1][k][word_id]);
            }
        }
        // calculation of constants for softmax transformation
        double constants = 0.0;
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            constants = logsumexp(constants, probs_n[k], (bool)(k == 0));
        }
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            probs_n[k] -= constants;
        }
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            probs_n[k] = exp(probs_n[k]);
        }
        // random sampling from multinomial distribution and assign new sense
        auto max_iter = std::max_element(probs_n, probs_n+trainer._scan->_n_k);
        int sense = std::distance(probs_n, max_iter);
        for (int i=0; i<trainer._scan->_context_window_width; ++i) {
            size_t word_id = dataset[n][i];
            if (word_id == trainer._vocab->num_words()) continue;
            if (trainer._word_frequency[word_id] < trainer._ignore_word_count) {
                continue;
            }
            log_pw += log(trainer._logistic_Psi[trainer._scan->_n_t-1][sense][word_id]);
        }
    }
    return log_pw;
}
int get_sum_word_frequency(SCANTrainer &trainer, unordered_map<size_t, int> word_frequency) {
    int sum = 0;
    for (int v=0; v<trainer._vocab->num_words(); ++v) {
        if (trainer._word_frequency[v] >= trainer._ignore_word_count) {
            sum += word_frequency[v];
        }
    }
    return sum;
}

// path to documents of time-step T + 1
DEFINE_string(data_path, "./data/transport/test.txt", "path to dataset for inference");
DEFINE_string(model_path, "./bin/transport.model", "path to model archive");

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    vector<vector<size_t>> dataset;
    unordered_map<size_t, int> word_frequency;
    SCANTrainer trainer;
    bool ret = trainer.load(FLAGS_model_path);
    trainer.initialize_cache();
    load_documents(FLAGS_data_path, trainer, dataset, word_frequency);
    double log_pw = compute_log_likelihood(trainer, dataset);
    double ppl = exp(-log_pw / get_sum_word_frequency(trainer, word_frequency));
    cout << "log_likelihood: " << log_pw << " perplexity: " << ppl << endl;
}
