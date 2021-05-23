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
std::vector<std::pair<wstring, double>> npmi_word_ranking(SCANTrainer &trainer, int t, int k) {
    std::vector<std::pair<wstring, double>> pw;
    int sum_word_frequency = trainer.get_sum_word_frequency();
    for (int v=0; v<trainer._scan->_vocab_size; ++v) {
        if (trainer._word_frequency[v] < trainer._ignore_word_count) {
            continue;
        }
        double ln_conditional_pw = log(trainer._logistic_Psi[t][k][v]);
        double ln_joint_pw = log(trainer._logistic_Psi[t][k][v]) + log(trainer._logistic_Phi[t][k]);
        double ln_pw = log((double)(trainer._word_frequency[v]) / (double)(sum_word_frequency));
        wstring word = trainer._vocab->word_id_to_string(v);
        pw.push_back(make_pair(word, (ln_conditional_pw - ln_pw) / -ln_joint_pw));
    }
    sort(pw.begin(), pw.end(), compare_by_b);
    return pw;
}
std::vector<double> sense_probability(SCANTrainer &trainer, int t) {
    std::vector<double> ps;
    for (int k=0; k<trainer._scan->_n_k; ++k) {
        ps.push_back(trainer._logistic_Phi[t][k]);
    }
    return ps;
}
std::vector<double> sense_gaussian(SCANTrainer &trainer, int t) {
    std::vector<double> ps;
    for (int k=0; k<trainer._scan->_n_k; ++k) {
        ps.push_back(trainer._scan->_Phi[t][k]);
    }
    return ps;
}
std::vector<std::pair<wstring, double>> marginal_word_ranking(SCANTrainer &trainer, int k) {
    std::unordered_map<int, double> id_prob;
    std::vector<std::pair<wstring, double>> pw;
    for (int t=0; t<trainer._scan->_n_t; ++t) {
        for (int v=0; v<trainer._scan->_vocab_size; ++v) {
            if (trainer._word_frequency[v] < trainer._ignore_word_count) {
                continue;
            }
            id_prob[v] += trainer._logistic_Psi[t][k][v];
        }
    }
    for (int v=0; v<trainer._scan->_vocab_size; ++v) {
        if (trainer._word_frequency[v] < trainer._ignore_word_count) {
            continue;
        }
        wstring word = trainer._vocab->word_id_to_string(v);
        pw.push_back(make_pair(word, id_prob[v]));
    }
    sort(pw.begin(), pw.end(), compare_by_b);
    return pw;
}
std::vector<std::pair<wstring, double>> npmi_marginal_word_ranking(SCANTrainer &trainer, int k) {
    std::unordered_map<int, double> id_prob;
    std::vector<std::pair<wstring, double>> pw;
    int sum_word_frequency = trainer.get_sum_word_frequency();
    for (int t=0; t<trainer._scan->_n_t; ++t) {
        for (int v=0; v<trainer._scan->_vocab_size; ++v) {
            if (trainer._word_frequency[v] < trainer._ignore_word_count) {
                continue;
            }
            id_prob[v] += trainer._logistic_Psi[t][k][v];
        }
    }
    for (int v=0; v<trainer._scan->_vocab_size; ++v) {
        if (trainer._word_frequency[v] < trainer._ignore_word_count) {
            continue;
        }
        double joint_pw = 0.0;
        for (int t=0; t<trainer._scan->_n_t; ++t) {
            joint_pw += trainer._logistic_Psi[t][k][v] * trainer._logistic_Phi[t][k];
        }
        double ln_conditional_pw = log(id_prob[v]);
        double ln_joint_pw = log(joint_pw);
        double ln_pw = log((double)(trainer._word_frequency[v]) / (double)(sum_word_frequency));
        wstring word = trainer._vocab->word_id_to_string(v);
        pw.push_back(make_pair(word, (ln_conditional_pw - ln_pw) / -ln_joint_pw));
    }
    sort(pw.begin(), pw.end(), compare_by_b);
    return pw;
}

DEFINE_string(model_path, "./bin/transport.model", "path to model archive");
DEFINE_bool(use_npmi, true, "use normalized pmi or not");
DEFINE_bool(normalize, true, "show lsb-normalized value or gaussian variable");

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
    // for each t, k, visualize word and sense distribution
    for (int t=0; t<trainer._scan->_n_t; ++t) {
        wcout << "time:" << t << endl;
        std::vector<double> ps;
        if (FLAGS_normalize) {
            ps = sense_probability(trainer, t);
        } else {
            ps = sense_gaussian(trainer, t);
        }
        for (int k=0; k<trainer._scan->_n_k; ++k) {
            wcout << ps[k] << " ";
            std::vector<std::pair<wstring, double>> pw;
            if (FLAGS_use_npmi) {
                pw = npmi_word_ranking(trainer, t, k);
            } else {
                pw = word_ranking(trainer, t, k);
            }
            for (int i=0; i<10; ++i) {
                wcout << pw[i].first << " ";
            }
            wcout << endl;
        }
    }
    wcout << "representative:" << endl;
    for (int k=0; k<trainer._scan->_n_k; ++k) {
        std::vector<std::pair<wstring, double>> mpw;
        if (FLAGS_use_npmi) {
            mpw = npmi_marginal_word_ranking(trainer, k);
        } else {
            mpw = marginal_word_ranking(trainer, k);
        }
        for (int i=0; i<10; ++i) {
            wcout << mpw[i].first << " ";
        }
        wcout << endl;
    }
    return 0;
}
