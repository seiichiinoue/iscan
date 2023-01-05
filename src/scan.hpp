#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <unordered_set>
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include "common.hpp"
#include "sampler.hpp"

using namespace std;

namespace scan {
    class SCAN {
    public:
        int _n_k;
        int _n_t;
        double _gamma_a;
        double _gamma_b;
        int _context_window_width;
        
        int _vocab_size;
        int _num_docs;

        // parameters
        double* _kappa_phi;
        double _kappa_psi;
        int* _Z;
        double** _Phi;
        double*** _Psi;
        double _scaling_coeff;

        SCAN() {
            _n_k = NUM_SENSE;
            _n_t = NUM_TIME;
            _gamma_a = GAMMA_A;
            _gamma_b = GAMMA_B;
            _context_window_width = CONTEXT_WINDOW_WIDTH;

            _vocab_size = 0;
            _num_docs = 0;

            _kappa_phi = NULL;
            _kappa_psi = KAPPA_PSI;
            _Z = NULL;
            _Phi = NULL;
            _Psi = NULL;
            _scaling_coeff = SCALING_COEFF;
        }
        ~SCAN() {
            if (_kappa_phi != NULL) {
                delete _kappa_phi;
            }
            if (_Z != NULL) {
                delete[] _Z;
            }
            if (_Phi != NULL) {
                for (int t=0; t<_n_t; ++t) {
                    if (_Phi[t] != NULL) {
                        delete[] _Phi[t];
                    }
                }
            }
            if (_Psi != NULL) {
                for (int t=0; t<_n_t; ++t) {
                    if (_Psi[t] != NULL) {
                        for (int k=0; k<_n_k; ++k) {
                            if (_Psi[t][k] != NULL) {
                                delete[] _Psi[t][k];
                            }
                        }
                    }
                }
            }
        }
        void initialize_cache(int num_time, int vocab_size, int num_docs) {
            _n_t = num_time;
            _vocab_size = vocab_size;
            _num_docs = num_docs;

            _kappa_phi = new double[_n_k];
            _Z = new int[num_docs];
            _Phi = new double*[_n_t];
            _Psi = new double**[_n_t];
        }
        void initialize_parameters(int word_identifier=0, double init_var=1.0) {
            for (int k=0; k<_n_k; ++k) {
                _kappa_phi[k] = KAPPA_PHI;
            }
            for (int n=0; n<_num_docs; ++n) {
                _Z[n] = sampler::uniform_int(0, _n_k-1);
            }
            for (int t=0; t<_n_t; ++t) {
                _Phi[t] = new double[_n_k];
                for (int k=0; k<_n_k; ++k) {
                    if (t == 0) {
                        _Phi[t][k] = 0.0;
                    } else {
                        _Phi[t][k] = _Phi[t-1][k] + generate_noise_from_normal_distribution() / sqrt(_kappa_phi[k]);
                    }
                }
            }
            for (int t=0; t<_n_t; ++t) {
                _Psi[t] = new double*[_n_k];
                for (int k=0; k<_n_k; ++k) {
                    _Psi[t][k] = new double[_vocab_size];
                    for (int v=0; v<_vocab_size; ++v) {
                        if (v == word_identifier) {
                            _Psi[t][k][v] = 0.0;
                        } else {
                            if (t == 0) {
                                _Psi[t][k][v] = generate_noise_from_normal_distribution() * init_var;
                            } else {
                                _Psi[t][k][v] = _Psi[t-1][k][v] + generate_noise_from_normal_distribution() / sqrt(_kappa_psi);
                            }
                        }
                    }
                }
            }
        }
        double generate_noise_from_normal_distribution() {
            return sampler::normal();
        }
        template<class Archive>
        void serialize(Archive &archive, unsigned int version) {
            boost::serialization::split_free(archive, *this, version);
        }
        void save(string filename) {
            std::ofstream ofs(filename);
            boost::archive::binary_oarchive oarchive(ofs);
            oarchive << *this;
        }
        bool load(string filename) {
            std::ifstream ifs(filename);
            if (ifs.good()) {
                boost::archive::binary_iarchive iarchive(ifs);
                iarchive >> *this;
                return true;
            }
            return false;
        }
    };
}
// save model
namespace boost { namespace serialization {
template<class Archive>
    void save(Archive &archive, const scan::SCAN &scan, unsigned int version) {
        archive & scan._n_k;
        archive & scan._n_t;
        archive & scan._gamma_a;
        archive & scan._gamma_b;
        archive & scan._context_window_width;
        archive & scan._vocab_size;
        archive & scan._num_docs;
        archive & scan._kappa_psi;
        archive & scan._scaling_coeff;
        for (int k=0; k<scan._n_k; ++k) {
            archive & scan._kappa_phi[k];
        }
        for (int n=0; n<scan._num_docs; ++n) {
            archive & scan._Z[n];
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                archive & scan._Phi[t][k];
            }
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                for (int v=0; v<scan._vocab_size; ++v) {
                    archive & scan._Psi[t][k][v];
                }
            }
        }
    }
template<class Archive>
    void load(Archive &archive, scan::SCAN &scan, unsigned int version) {
        archive & scan._n_k;
        archive & scan._n_t;
        archive & scan._gamma_a;
        archive & scan._gamma_b;
        archive & scan._context_window_width;
        archive & scan._vocab_size;
        archive & scan._num_docs;
        archive & scan._kappa_psi;
        archive & scan._scaling_coeff;
        if (scan._kappa_phi == NULL) {
            scan._kappa_phi = new double[scan._n_k];
        }
        if (scan._Z == NULL) {
            scan._Z = new int[scan._num_docs];
        }
        if (scan._Phi == NULL) {
            scan._Phi = new double*[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._Phi[t] = new double[scan._n_k];
            }
        }
        if (scan._Psi == NULL) {
            scan._Psi = new double**[scan._n_t];
            for (int t=0; t<scan._n_t; ++t) {
                scan._Psi[t] = new double*[scan._n_k];
                for (int k=0; k<scan._n_k; ++k) {
                    scan._Psi[t][k] = new double[scan._vocab_size];
                }
            }
        }
        for (int k=0; k<scan._n_k; ++k) {
            archive & scan._kappa_phi[k];
        }
        for (int n=0; n<scan._num_docs; ++n) {
            archive & scan._Z[n];
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                archive & scan._Phi[t][k];
            }
        }
        for (int t=0; t<scan._n_t; ++t) {
            for (int k=0; k<scan._n_k; ++k) {
                for (int v=0; v<scan._vocab_size; ++v) {
                    archive & scan._Psi[t][k][v];
                }
            }
        }
    }
}}  // namespace boost::serialization