#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <cstdlib>
#include <thread>
#include <dirent.h>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map> 
#include "scan.hpp"
#include "vocab.hpp"
using namespace boost;
using namespace scan;

template<typename T>
struct multiset_comparator {
    bool operator()(const pair<size_t, T> &a, const pair<size_t, T> &b) {
        return a.second > b.second;
    }
};

bool compare(const pair<size_t, int> &a, const pair<size_t, int> &b) {
    return a.second > b.second;
}
void split_string_by(const string &str, char delim, vector<string> &elems) {
    elems.clear();
    string item;
    for (char ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}
void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems) {
    elems.clear();
    wstring item;
    for (wchar_t ch : str) {
        if (ch == delim) {
            if (!item.empty()) {
                elems.push_back(item);
            }
            item.clear();
        } else {
            item += ch;
        }
    }
    if (!item.empty()) {
        elems.push_back(item);
    }
}
bool ends_with(const std::string& str, const std::string& suffix) {
    size_t len1 = str.size();
    size_t len2 = suffix.size();
    return len1 >= len2 && str.compare(len1 - len2, len2, suffix) == 0;
}

class SCANTrainer {
public:
    SCAN *_scan;
    Vocab *_vocab;
    vector<vector<size_t>> _dataset;
    vector<int> _times;
    unordered_map<size_t, int> _word_frequency;
    unordered_map<int, int> _snippet_count;

    double** _logistic_Phi;
    double*** _logistic_Psi;
    double* _probs;
    double* _prior_mean_phi;
    double* _prior_sigma2_phi;
    double* _prior_mean_psi;

    int _start_year;
    int _end_year;
    int _year_interval;

    int _burn_in_period;
    int _top_n_word;
    int _min_word_count;
    int _min_snippet_count;
    int _min_snippet_length;
    int _kappa_phi_interval;
    int _kappa_phi_start;
    int _current_iter;

    double _sigma_coeff;
    normal_distribution<double> _noise_coeff;
    
    SCANTrainer() {
        setlocale(LC_CTYPE, "ja_JP.UTF-8");
        ios_base::sync_with_stdio(false);
        locale default_loc("ja_JP.UTF-8");
        locale::global(default_loc);
        locale ctype_default(locale::classic(), default_loc, locale::ctype);
        wcout.imbue(ctype_default);
        wcin.imbue(ctype_default);

        _scan = new SCAN();
        _vocab = new Vocab();
        _logistic_Phi = NULL;
        _logistic_Psi = NULL;
        _probs = NULL;
        _prior_mean_phi = NULL;
        _prior_sigma2_phi = NULL;
        _prior_mean_psi = NULL;

        _sigma_coeff = SIGMA_COEFF;

        _start_year = START_YEAR;
        _end_year = END_YEAR;
        _year_interval = YEAR_INTERVAL;

        _burn_in_period = BURN_IN_PERIOD;
        _top_n_word = TOP_N_WORD;
        _min_word_count = MIN_WORD_COUNT;
        _min_snippet_count = MIN_SNIPPET_COUNT;
        _min_snippet_length = MIN_SNIPPET_LENGTH;
        _kappa_phi_start = KAPPA_PHI_START;
        _kappa_phi_interval = KAPPA_PHI_INTERVAL;
        _current_iter = 0;
    }
    ~SCANTrainer() {
        if (_logistic_Phi != NULL) {
            for (int t=0; t<_scan->_n_t; ++t) {
                if (_logistic_Phi[t] != NULL) {
                    delete[] _logistic_Phi[t];
                }
            }
        }
        if (_logistic_Psi != NULL) {
            for (int t=0; t<_scan->_n_t; ++t) {
                if (_logistic_Psi[t] != NULL) {
                    for (int k=0; k<_scan->_n_k; ++k) {
                        if (_logistic_Psi[t][k] != NULL) {
                            delete[] _logistic_Psi[t][k];
                        }
                    }
                }
            }
        }
        if (_probs != NULL) {
            delete[] _probs;
        }
        if (_prior_mean_phi != NULL) {
            delete[] _prior_mean_phi;
        }
        if (_prior_sigma2_phi != NULL) {
            delete[] _prior_sigma2_phi;
        }
        if (_prior_mean_psi != NULL) {
            delete[] _prior_mean_psi;
        }
        delete _scan;
        delete _vocab;
    }
    void load_documents(string filepath) {
        unordered_map<int, int> year_to_id;
        for (int y=_start_year; y<=_end_year; ++y) {
            year_to_id[y] = (y - _start_year) / _year_interval;
        }
        wifstream ifs(filepath.c_str());
        assert(ifs.fail() == false);
        wstring sentence;
        while (getline(ifs, sentence) && !ifs.eof()) {
            vector<wstring> words;
            split_word_by(sentence, L' ', words);
            if (words.size() - 1 < _min_snippet_length) {
                continue;
            }
            int doc_id = _dataset.size();
            _dataset.push_back(vector<size_t>());
            int time_id = year_to_id[stoi(words[0])];
            words.erase(words.begin());
            _add_document(words, doc_id);
            _times.push_back(time_id);
            _snippet_count[time_id] += 1;
        }
    }
    void _add_document(vector<wstring> &words, int doc_id) {
        if (words.size() == 0) return;
        vector<size_t> &doc = _dataset[doc_id];
        for (auto word : words) {
            if (word.size() == 0) continue;
            size_t word_id = _vocab->add_string(word);
            doc.push_back(word_id);
            _word_frequency[word_id] += 1;
        }
    }
    void _initialize_parameters() {
        for (int t=0; t<_scan->_n_t; ++t) {
            vector<int> cnt_t(_scan->_n_k, 0);
            int sum_cnt_t = 0;
            for (int n=0; n<_scan->_num_docs; ++n) {
                if (_times[n] != t) continue;
                cnt_t[_scan->_Z[n]]++;
                sum_cnt_t++;
            }
            // initialize Phi with MLE
            for (int k=0; k<_scan->_n_k; ++k) {
                _scan->_Phi[t][k] = ((double)cnt_t[k] + 0.01) / ((double)sum_cnt_t + (_scan->_n_k * 0.01));
            }
            for (int k=0; k<_scan->_n_k; ++k) {
                vector<int> cnt_t_k(_scan->_vocab_size, 0);
                int sum_cnt_t_k = 0;
                for (int n=0; n<_scan->_num_docs; ++n) {
                    if (_times[n] != t || _scan->_Z[n] != k) continue;
                    for (int i=0; i<_dataset[n].size(); ++i) {
                        size_t word_id = _dataset[n][i];
                        if (_word_frequency[word_id] < _min_word_count) {
                            continue;
                        }
                        cnt_t_k[word_id]++;
                        sum_cnt_t_k++;
                    }
                }
                // initialize Psi with MLE
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _scan->_Psi[t][k][v] = ((double)cnt_t_k[v] + 0.01) / ((double)sum_cnt_t_k + (_scan->_vocab_size * 0.01));
                }
            }
        }
    }
    void _compute_min_word_count() {
        vector<pair<size_t, int>> ordered_vocab(_word_frequency.begin(), _word_frequency.end());
        sort(ordered_vocab.begin(), ordered_vocab.end(), compare);
        if (ordered_vocab.size() > _top_n_word) {
            _min_word_count = ordered_vocab[_top_n_word-1].second;
        } else {
            _min_word_count = 0;
        }
    }
    void initialize_cache() {
        _logistic_Phi = new double*[_scan->_n_t];
        _logistic_Psi = new double**[_scan->_n_t];
        _probs = new double[_scan->_n_k];
        _prior_mean_phi = new double[_scan->_n_k];
        _prior_sigma2_phi = new double[_scan->_n_k];
        _prior_mean_psi = new double[_scan->_vocab_size];

        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Phi[t] = new double[_scan->_n_k];
            _stick_breaking_transformation(t, _logistic_Phi[t]);
        }
        for (int t=0; t<_scan->_n_t; ++t) {
            _logistic_Psi[t] = new double*[_scan->_n_k];
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_Psi[t][k] = new double[_scan->_vocab_size];
                _logistic_transformation(t, k, _logistic_Psi[t][k]);
            }
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            _probs[k] = 0.0;
        }
    }
    void prepare(bool mle=false) {
        int num_time = ((_end_year - _start_year) + (_year_interval - 1)) / _year_interval;
        int vocab_size = _vocab->num_words();
        int num_docs = _dataset.size();
        _scan->initialize_cache(num_time, vocab_size, num_docs);
        // initialize parameters $\phi$ and $\psi$ with MLE
        if (mle) {
            _initialize_parameters();
        }
        // after initializing $\phi$ and $\psi$, initialize trainer's chache
        initialize_cache();
        // initialize gaussian sampler for MH sampling
        _noise_coeff = normal_distribution<double>(0, _sigma_coeff);
        // compute min_word_count according to top_n_word
        if (_min_word_count == 0) {  // initial value
            _compute_min_word_count();
        }
    }
    void set_num_sense(int n_k) {
        _scan->_n_k = n_k;
    }
    void set_kappa_phi(double kappa_phi) {
        for (int k=0; k<_scan->_n_k; ++k) {
            _scan->_kappa_phi[k] = kappa_phi;
        }
    }
    void set_kappa_psi(double kappa_psi) {
        _scan->_kappa_psi = kappa_psi;
    }
    void set_gamma_a(double gamma_a) {
        _scan->_gamma_a = gamma_a;
    }
    void set_gamma_b(double gamma_b) {
        _scan->_gamma_b = gamma_b;
    }
    void set_kappa_phi_start(int kappa_phi_start) {
        _kappa_phi_start = kappa_phi_start;
    }
    void set_kappa_phi_interval(int kappa_phi_interval) {
        _kappa_phi_interval = kappa_phi_interval;
    }
    void set_scaling_coeff(double scaling_coeff) {
        _scan->_scaling_coeff = scaling_coeff;
    }
    void set_sigma_coeff(double sigma_coeff) {
        _sigma_coeff = sigma_coeff;
    }
    void set_context_window_width(int context_window_width) {
        _scan->_context_window_width = context_window_width;
    }
    void set_start_year(int start_year) {
        _start_year = start_year;
    }
    void set_end_year(int end_year) {
        _end_year = end_year;
    }
    void set_year_interval(int year_interval) {
        _year_interval = year_interval;
    }
    void set_burn_in_period(int burn_in_period) {
        _burn_in_period = burn_in_period;
    }
    void set_top_n_word(int top_n_word) {
        _top_n_word = top_n_word;
    }
    void set_min_word_count(int min_word_count) {
        _min_word_count = min_word_count;
    }
    void set_min_snippet_count(int min_snippet_count) {
        _min_snippet_count = min_snippet_count;
    }
    void set_min_snippet_length(int min_snippet_length) {
        _min_snippet_length = min_snippet_length;
    }
    int get_sum_word_frequency() {
        int sum = 0;
        for (int v=0; v<_vocab->num_words(); ++v) {
            if (_word_frequency[v] >= _min_word_count) {
                sum += _word_frequency[v];
            }
        }
        return sum;
    }
    int get_vocab_size() {
        return _vocab->num_words() - get_ignore_vocab_size();
    }
    int get_original_vocab_size() {
        return _vocab->num_words();
    }
    int get_ignore_vocab_size() {
        int cnt = 0;
        for (int v=0; v<_word_frequency.size(); ++v) {
            if (_word_frequency[v] < _min_word_count) {
                cnt++;
            }
        }
        return cnt;
    }
    void sample_z(int t) {
        _update_logistic_Phi();
        _update_logistic_Psi();
        double** logistic_psi_t = _logistic_Psi[t];
        double* logistic_phi_t = _logistic_Phi[t];
        for (int n=0; n<_scan->_num_docs; ++n) {
            if (_times[n] != t) continue;
            // calculation for $\phi^t_k$
            double* probs_n = _probs;
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] = log(logistic_phi_t[k]);
            }
            // calculation for $\prod \psi^{t, k}_wi$
            for (int k=0; k<_scan->_n_k; ++k) {
                for (int i=0; i<_dataset[n].size(); ++i) {
                    size_t word_id = _dataset[n][i];
                    if (_word_frequency[word_id] < _min_word_count) {
                        continue;
                    }
                    probs_n[k] += log(logistic_psi_t[k][word_id]);
                }
            }
            // calculation of constants for softmax transformation
            double constants = 0.0;
            for (int k=0; k<_scan->_n_k; ++k) {
                constants = logsumexp(constants, probs_n[k], (bool)(k == 0));
            }
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] -= constants;
            }
            for (int k=0; k<_scan->_n_k; ++k) {
                probs_n[k] = exp(probs_n[k]);
            }
            // random sampling from multinomial distribution and assign new sense
            int sense = sampler::multinomial((size_t)_scan->_n_k, probs_n);
            _scan->_Z[n] = sense;
        }
    }
    void sample_phi(int t) {
        // sample phi under each time $t$
        _update_logistic_Phi();
        double* phi_t = _scan->_Phi[t];
        double* logistic_phi_t = _logistic_Phi[t];
        if (t == 0) {
            for (int k=0; k<_scan->_n_k; ++k) {
                _prior_mean_phi[k] = _scan->_Phi[t+1][k];
                _prior_sigma2_phi[k] = 1.0 / _scan->_kappa_phi[k];
            }
        } else if (t+1 == _scan->_n_t) {
            for (int k=0; k<_scan->_n_k; ++k) {
                _prior_mean_phi[k] = _scan->_Phi[t-1][k];
                _prior_sigma2_phi[k] = 1.0 / _scan->_kappa_phi[k];
            }
        } else {
            for (int k=0; k<_scan->_n_k; ++k) {
                _prior_mean_phi[k] = _scan->_Phi[t-1][k] + _scan->_Phi[t+1][k];
                _prior_mean_phi[k] *= 0.5;
                _prior_sigma2_phi[k] = 1.0 / (2.0 * _scan->_kappa_phi[k]);
            }
        }
        vector<int> cnt_t(_scan->_n_k, 0);
        int sum_cnt_t = 0;
        for (int n=0; n<_scan->_num_docs; ++n) {
            if (_times[n] != t) continue;
            cnt_t[_scan->_Z[n]]++;
            sum_cnt_t++;
        }
        vector<int> nx(_scan->_n_k-1, 0);
        int cnt = 0;
        for (int k=0; k<_scan->_n_k-1; ++k) {
            nx[k] = sum_cnt_t - cnt;
            cnt += cnt_t[k];
        }
        // sampling with polya-gamma sampler
        for (int k=0; k<_scan->_n_k-1; ++k) {
            double omega_k = sampler::polya_gamma(nx[k], phi_t[k]);
            double sigma2_k_tilde = (double)(1.0) / (omega_k + ((double)(1.0) / _prior_sigma2_phi[k]));
            double mu_k_tilde = ((cnt_t[k] - (nx[k] / (double)(2.0))) + (_prior_mean_phi[k] / _prior_sigma2_phi[k])) * sigma2_k_tilde;
            double noise = sampler::normal();
            double sampled = mu_k_tilde + noise * sqrt(sigma2_k_tilde);
            _scan->_Phi[t][k] = sampled;
        }
        // sanity check
        double sum = 0;
        for (int k=0; k<_scan->_n_k; ++k) {
            sum += _logistic_Phi[t][k];
        }
        assert(abs(1.0 - sum) < 1e-5);
        return;
    }
    void sample_psi(int t) {
        // sample phi under each {time $t$, sense $k$}
        _update_logistic_Psi();
        for (int k=0; k<_scan->_n_k; ++k) {
            double prior_sigma;
            double* psi_t_k = _scan->_Psi[t][k];
            double* logistic_psi_t_k = _logistic_Psi[t][k];
            if (t == 0) {
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _prior_mean_psi[v] = _scan->_Psi[t+1][k][v];
                }
                prior_sigma = sqrt(1.0 / _scan->_kappa_psi);
            } else if (t+1 == _scan->_n_t) {
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _prior_mean_psi[v] = _scan->_Psi[t-1][k][v];
                }
                prior_sigma = sqrt(1.0 / _scan->_kappa_psi);
            } else {
                for (int v=0; v<_scan->_vocab_size; ++v) {
                    _prior_mean_psi[v] = _scan->_Psi[t-1][k][v] + _scan->_Psi[t+1][k][v];
                    _prior_mean_psi[v] *= 0.5;
                }
                prior_sigma = sqrt(1.0 / (2.0 * _scan->_kappa_psi));
            }
            vector<int> cnt_t_k(_scan->_vocab_size, 0);
            int sum_cnt_t_k = 0;
            for (int n=0; n<_scan->_num_docs; ++n) {
                if (_times[n] != t || _scan->_Z[n] != k) continue;
                for (int i=0; i<_dataset[n].size(); ++i) {
                    size_t word_id = _dataset[n][i];
                    if (_word_frequency[word_id] < _min_word_count) {
                        continue;
                    }
                    cnt_t_k[word_id]++;
                    sum_cnt_t_k++;
                }
            }
            double denom = 0.0;
            for (int v=0; v<_scan->_vocab_size; ++v) {
                if (_word_frequency[v] < _min_word_count) {
                    continue;
                }
                denom += exp(psi_t_k[v]);
            }
            for (int v=0; v<_scan->_vocab_size; ++v) {
                if (_word_frequency[v] < _min_word_count) {
                    continue;
                }
                double constants = denom - exp(psi_t_k[v]);
                int cnt = cnt_t_k[v];
                int cnt_else = sum_cnt_t_k - cnt_t_k[v];
                double lu, ru;
                // random sampling of maximum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(0, logistic_psi_t_k[v])$ 
                lu = std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt) * logistic_psi_t_k[v];
                lu = log(constants) + log(lu) - log(1.0 - lu);
                // random sampling of minimum value in $log(u_n / (1 - u_n))$, where $u_n \sim U(logistic_psi_t_k[v], 1)$
                ru = (1.0 - logistic_psi_t_k[v]) * (1.0 - std::pow(sampler::uniform(0, 1), 1.0 / (double)cnt_else)) + logistic_psi_t_k[v];
                ru = log(constants) + log(ru) - log(1.0 - ru);
                // scaling probabilistic variable to standard normal
                lu = (lu - _prior_mean_psi[v]) / prior_sigma;
                ru = (ru - _prior_mean_psi[v]) / prior_sigma;
                assert(lu < ru);
                double noise = sampler::truncated_normal(lu, ru);
                double sampled = _prior_mean_psi[v] + noise * prior_sigma;
                _scan->_Psi[t][k][v] = sampled;
            }
            // sanity check
            double sum = 0;
            for (int v=0; v<_scan->_vocab_size; ++v) {
                if (_word_frequency[v] < _min_word_count) continue;
                sum += _logistic_Psi[t][k][v];
            }
            assert(abs(1.0 - sum) < 1e-5);
        }
        return;
    }
    void sample_kappa() {
        double a = _scan->_gamma_a + (double)(_scan->_n_t) * 0.5;
        for (int k=0; k<_scan->_n_k-1; ++k) {
            double b = 0.0;
            double mu_phi = 0.0;
            for (int t=0; t<_scan->_n_t; ++t) {
                mu_phi += _scan->_Phi[t][k];
            }
            mu_phi /= (double)_scan->_n_t;
            for (int t=0; t<_scan->_n_t; ++t) {
                b += pow(_scan->_Phi[t][k] - mu_phi, 2);
            }
            b = _scan->_gamma_b + (b / 2.0);
            _scan->_kappa_phi[k] = sampler::gamma(a, b);
        }
        return;
    }
    bool sample_scaling_coeff() {
        double scaling_coeff_old = _scan->_scaling_coeff;
        double z = _noise_coeff(sampler::minstd);
        double scaling_coeff_new = scaling_coeff_old * exp(z);
        // compute log-likelihood for scaling_coeff_old
        double log_pw_old = compute_log_likelihood();
        // set new sampled parameter
        _scan->_scaling_coeff = scaling_coeff_new;
        // compute log-likelihood for scaling_coeff_new
        double log_pw_new = compute_log_likelihood();
        // if accept; update scaling_coeff parameter
        double log_acceptance_rate = log_pw_new - log_pw_old;
        double acceptance_ratio = std::min(1.0, exp(log_acceptance_rate));
        double bernoulli = sampler::uniform(0, 1);
        if (bernoulli <= acceptance_ratio) {
            return true;
        }
        // else; undo
        _scan->_scaling_coeff = scaling_coeff_old;
        _update_logistic_Phi();
        return false;
    }
    bool _word_in_document(size_t word_id, int doc_id) {
        vector<size_t>& tar_doc = _dataset[doc_id];
        for (int i=0; i<tar_doc.size(); ++i) {
            if (tar_doc[i] == word_id) return true;
        }
        return false;
    }
    void _update_logistic_Phi() {
        for (int t=0; t<_scan->_n_t; ++t) {
            _stick_breaking_transformation(t, _logistic_Phi[t]);
        }
    }
    void _update_logistic_Psi() {
        for (int t=0; t<_scan->_n_t; ++t) {
            for (int k=0; k<_scan->_n_k; ++k) {
                _logistic_transformation(t, k, _logistic_Psi[t][k]);
            }
        }
    }
    void _logistic_transformation(int t, double* vec) {
        double* phi_t = _scan->_Phi[t];
        double u = 0.0;
        for (int k=0; k<_scan->_n_k; ++k) {
            u = logsumexp(u, phi_t[k], (bool)(k == 0));
        }
        for (int k=0; k<_scan->_n_k; ++k) {
            vec[k] = exp(phi_t[k] - u);
        }
    }
    void _logistic_transformation(int t, int k, double* vec) {
        double* psi_t_k = _scan->_Psi[t][k];
        double u = 0.0;
        bool init_flag = 1;
        for (int v=0; v<_scan->_vocab_size; ++v) {
            if (_word_frequency[v] < _min_word_count) {
                continue;
            }
            u = logsumexp(u, psi_t_k[v], init_flag);
            if (init_flag) init_flag = 0;
        }
        for (int v=0; v<_scan->_vocab_size; ++v) {
            vec[v] = exp(psi_t_k[v] - u);
        }
    }
    void _stick_breaking_transformation(int t, double* vec) {
        double* phi_t = _scan->_Phi[t];
        double stick = 1.0;
        for (int k=0; k<_scan->_n_k-1; ++k) {
            vec[k] = logistic(phi_t[k] * _scan->_scaling_coeff) * stick;
            stick -= vec[k];
        }
        vec[_scan->_n_k-1] = stick;
    }
    double compute_log_likelihood() {
        _update_logistic_Phi();
        _update_logistic_Psi();
        double log_pw = 0.0;
        for (int t=0; t<_scan->_n_t; ++t) {
            for (int n=0; n<_scan->_num_docs; ++n) {
                double log_pw_d = 0.0;
                if (_times[n] != t) continue;
                for (int k=0; k<_scan->_n_k-1; ++k) {
                    double log_pw_dk = log(_logistic_Phi[t][k]);
                    for (int i=0; i<_dataset[n].size(); ++i) {
                        size_t word_id = _dataset[n][i];
                        if (_word_frequency[word_id] < _min_word_count) {
                            continue;
                        }
                        log_pw_dk += log(_logistic_Psi[t][k][word_id]);
                    }
                    log_pw_d += exp(log_pw_dk);
                }
                log_pw += log(log_pw_d);
            }
        }
        return log_pw;
    }
    void train(int iter=1000, string save_path ="./bin/scan.model") {
        for (int i=0; i<iter; ++i) {
            ++_current_iter;
            for (int t=0; t<_scan->_n_t; ++t) {
                sample_z(t);
                sample_phi(t);
                sample_psi(t);
            }
            if (_current_iter > _kappa_phi_start && _current_iter % _kappa_phi_interval == 0) {
                sample_kappa();
            }
            double log_pw = compute_log_likelihood();
            double ppl = exp((-1 * log_pw) / get_sum_word_frequency());
            cout << "iter: " << _current_iter 
                << " log_likelihood: " << log_pw
                << " perplexity: " << ppl 
                << " kappa_phi: ";
            for (int k=0; k<_scan->_n_k-1; ++k) {
                cout << _scan->_kappa_phi[k] << ",";
            }
            cout << _scan->_kappa_phi[_scan->_n_k-1] << endl;
            save(save_path);
        }
    }
    void save(string filename) {
        std::ofstream ofs(filename);
        boost::archive::binary_oarchive oarchive(ofs);
        oarchive << *_vocab;
        oarchive << *_scan;
        oarchive << _word_frequency;
        oarchive << _dataset;
        oarchive << _times;
        oarchive << _start_year;
        oarchive << _end_year;
        oarchive << _year_interval;
        oarchive << _burn_in_period;
        oarchive << _top_n_word;
        oarchive << _min_word_count;
        oarchive << _min_snippet_count;
        oarchive << _min_snippet_length;
        oarchive << _kappa_phi_start;
        oarchive << _kappa_phi_interval;
        oarchive << _current_iter;
        oarchive << _sigma_coeff;
    }
    bool load(string filename) {
        std::ifstream ifs(filename);
        if (ifs.good()) {
            _vocab = new Vocab();
            _scan = new SCAN();
            boost::archive::binary_iarchive iarchive(ifs);
            iarchive >> *_vocab;
            iarchive >> *_scan;
            iarchive >> _word_frequency;
            iarchive >> _dataset;
            iarchive >> _times;
            iarchive >> _start_year;
            iarchive >> _end_year;
            iarchive >> _year_interval;
            iarchive >> _burn_in_period;
            iarchive >> _top_n_word;
            iarchive >> _min_word_count;
            iarchive >> _min_snippet_count;
            iarchive >> _min_snippet_length;
            iarchive >> _kappa_phi_start;
            iarchive >> _kappa_phi_interval;
            iarchive >> _current_iter;
            iarchive >> _sigma_coeff;
            return true;
        }
        return false;
    }
};
