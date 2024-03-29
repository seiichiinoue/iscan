#include "src/model.hpp"

// hyper parameters flags
DEFINE_int32(num_sense, 8, "number of sense");
DEFINE_double(kappa_phi, 4.0, "initial value of kappa_phi");
DEFINE_double(kappa_psi, 100.0, "initial value of kappa_psi (fixed)");
DEFINE_double(gamma_a, 7.0, "hyperparameter of gamma prior");
DEFINE_double(gamma_b, 3.0, "hyperparameter of gamma prior");
DEFINE_int32(kappa_phi_start, 100, "start point of kappa sampling");
DEFINE_int32(kappa_phi_interval, 50, "interval of kappa sampling");
DEFINE_double(scaling_coeff, 1.0, "concentration parameter of LSBP");
DEFINE_double(sigma_coeff, 0.05, "random walk width of MH sampling");
DEFINE_int32(start_year, 1700, "start year in the corpus");
DEFINE_int32(end_year, 2020, "end year in the corpus");
DEFINE_int32(year_interval, 20, "year interval");
DEFINE_int32(context_window_width, 5, "context window width");
DEFINE_int32(num_iteration, 1000, "number of iteration");
DEFINE_int32(burn_in_period, 500, "burn in period");
DEFINE_int32(top_n_word, 3000, "threshold for vocabulary selection");
DEFINE_int32(min_word_count, -1, "threshold of low-frequency words");
DEFINE_int32(min_snippet_count, 1, "threshold for snippets size in the time point");
DEFINE_int32(min_snippet_length, 1, "threshold of size of snippet");
DEFINE_string(data_path, "./data/transport/corpus.txt", "path to dataset for training");
DEFINE_string(save_path, "./bin/scan.model", "path to model for archive");
DEFINE_string(load_path, "./bin/scan.model", "path to model for loading");
DEFINE_bool(from_archive, false, "load archive or not");
DEFINE_bool(use_initial_variance, false, "use initial variance for initialization of word distribution");
DEFINE_bool(mle_initialize, false, "use mle for parameter initialization or not");

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(*argv);
    google::ParseCommandLineFlags(&argc, &argv, true);
    SCANTrainer trainer;
    // set hyper parameters
    trainer.set_num_sense(FLAGS_num_sense);
    trainer.set_kappa_phi_start(FLAGS_kappa_phi_start);
    trainer.set_kappa_phi_interval(FLAGS_kappa_phi_interval);
    trainer.set_scaling_coeff(FLAGS_scaling_coeff);
    trainer.set_sigma_coeff(FLAGS_sigma_coeff);
    trainer.set_start_year(FLAGS_start_year);
    trainer.set_end_year(FLAGS_end_year);
    trainer.set_year_interval(FLAGS_year_interval);
    trainer.set_context_window_width(FLAGS_context_window_width);
    trainer.set_burn_in_period(FLAGS_burn_in_period);
    trainer.set_top_n_word(FLAGS_top_n_word);
    trainer.set_min_word_count(FLAGS_min_word_count);
    trainer.set_min_snippet_count(FLAGS_min_snippet_count);
    trainer.set_min_snippet_length(FLAGS_min_snippet_length);
    // load dataset
    trainer.load_documents(FLAGS_data_path);
    // prepare model
    trainer.prepare(FLAGS_use_initial_variance, FLAGS_mle_initialize);
    // initialize parameters
    trainer.set_kappa_phi(FLAGS_kappa_phi);
    trainer.set_kappa_psi(FLAGS_kappa_psi);
    trainer.set_gamma_a(FLAGS_gamma_a);
    trainer.set_gamma_b(FLAGS_gamma_b);
    // load archive if from_archive is true
    if (FLAGS_from_archive) {
        trainer.load(FLAGS_load_path);
        cout << "model loaded from archive: " << FLAGS_load_path << endl;
        cout << "starting training at iter: " << trainer._current_iter + 1 << endl;
    }
    // logging summary
    cout << "{num_sense: " << trainer._scan->_n_k << ", num_time: " << trainer._scan->_n_t
        << ", kappa_psi: " << trainer._scan->_kappa_psi
        << ", gamma_a: " << trainer._scan->_gamma_a << ", gamma_b: " << trainer._scan->_gamma_b
        << ", scaling_coeff: " << trainer._scan->_scaling_coeff
        << ", num_iteration: " << FLAGS_num_iteration
        << ", min_word_count: " << FLAGS_min_word_count
        << ", top_n_word: " << FLAGS_top_n_word
        << ", min_snippet_count: " << FLAGS_min_snippet_count
        << ", min_snippet_length: " << FLAGS_min_snippet_length << "}" << endl;
    cout << "num of docs: " << trainer._scan->_num_docs << endl;
    cout << "sum of word freq: " << trainer.get_sum_word_frequency() << endl;
    cout << "vocab size: " << trainer.get_vocab_size() << endl;
    cout << "original vocab size: " << trainer.get_original_vocab_size() << endl;
    cout << "ignore vocab size: " << trainer.get_ignore_vocab_size() << endl;
    // tarining
    trainer.train(FLAGS_num_iteration, FLAGS_save_path);
    return 0;
}
