#include "src/model.hpp"

bool compare_by_b(std::pair<wstring, double> a, std::pair<wstring, double> b) {
    if (a.second != b.second) {
        return a.second > b.second;
    } else {
        return a.first < b.first;
    }
}
std::vector<std::pair<wstring, double>> word_ranking(SCANTrainer &trainer, int t, int k) {
    std::vector<std::pair<wstring, double>> pw;
    for (int v=0; v<trainer._scan->_vocab_size; ++v) {
        if (trainer._word_frequency[v] < trainer._ignore_word_count) {
            continue;
        }
        wstring word = trainer._vocab->word_id_to_string(v);
        pw.push_back(make_pair(word, trainer._logistic_Psi[t][k][v]));
    }
    sort(pw.begin(), pw.end(), compare_by_b);
    return pw;
}
// calculate coherence with u_mass way
int document_frequency(SCANTrainer &trainer, size_t wi) {
    int cnt = 0;
    for (int doc_id=0; doc_id<trainer._scan->_num_docs; ++doc_id) {
        vector<size_t>& tar_doc = trainer._dataset[doc_id];
        for (int i=0; i<tar_doc.size(); ++i) {
            if (tar_doc[i] == wi) {
                cnt++;
                break;
            }
        }
    }
    return cnt;
}
int bigram_document_grequency(SCANTrainer &trainer, size_t wi, size_t wj) {
    int cnt = 0;
    for (int doc_id=0; doc_id<trainer._scan->_num_docs; ++doc_id) {
        vector<size_t>& tar_doc = trainer._dataset[doc_id];
        bool flag_i = false, flag_j = false;
        for (int i=0; i<tar_doc.size(); ++i) {
            if (tar_doc[i] == wi) {
                flag_i = true;
            }
            if (tar_doc[i] == wj) {
                flag_j = true;
            }
        }
        if (flag_i && flag_j) {
            cnt++;
        }
    }
    return cnt;
}
double coherence(SCANTrainer &trainer) {
    double score = 0.0;
    for (int t=0; t<trainer._scan->_n_t; ++t) {
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            std::vector<std::pair<wstring, double>> top10 = word_ranking(trainer, t, k);
            double tmp = 0.0;
            for (int i=0; i<10; ++i) {
                int wi = trainer._vocab->get_word_id(top10[i].first);
                int freq_i = document_frequency(trainer, wi);
                for (int j=i+1; j<10; ++j) {
                    size_t wj = trainer._vocab->get_word_id(top10[j].first);
                    int freq_j = document_frequency(trainer, wj);
                    int freq_ij = bigram_document_grequency(trainer, wi, wj);
                    double f_ij;
                    if (freq_ij == 0) {
                        f_ij = -1;
                    } else {
                        f_ij = -1 + (log(freq_i) + log(freq_j) - 2.0 * log(trainer._scan->_num_docs)) / (log(freq_ij) - log(trainer._scan->_num_docs));
                    }
                    tmp += f_ij;
                }
            }
            score += tmp / 45.0;
        }
    }
    return score / (trainer._scan->_n_t * trainer._scan->_n_k);
}

DEFINE_string(model_path, "./bin/transport.model", "path to model archive");

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    SCANTrainer trainer;
    // load archive
    bool ret = trainer.load(FLAGS_model_path);
    // prepare model
    trainer.initialize_cache();
    trainer._update_logistic_Phi();
    trainer._update_logistic_Psi();
    // calculate coherence
    double ch = coherence(trainer);
    cout << "coherence: " << ch << endl;
    return 0;
}
