#pragma once
#include <random>
#include <chrono>
#include <boost/math/special_functions/erf.hpp>

using namespace std;

namespace scan {
    namespace sampler {
        int seed = chrono::system_clock::now().time_since_epoch().count();
        mt19937 mt(seed);
        minstd_rand minstd(seed);

        double gamma(double a, double b) {
            gamma_distribution<double> distribution(a, 1.0 / b);
            return distribution(mt);
        }
        double beta(double a, double b) {
            double ga = gamma(a, 1.0);
            double gb = gamma(b, 1.0);
            return ga / (ga + gb);
        }
        double bernoulli(double p) {
            uniform_real_distribution<double> rand(0, 1);
            double r = rand(mt);
            if (r > p) {
                return 0;
            }
            return 1;
        }
        double uniform(double min=0, double max=0) {
            uniform_real_distribution<double> rand(min, max);
            return rand(mt);
        }
        static double uniform_int(int min=0, int max=0) {
            uniform_int_distribution<> rand(min, max);
            return rand(mt);
        }
        double _normal_cdf(double x) {
            return 0.5 * erfc(-x * std::sqrt(0.5));
        }
        double _inverse_normal_cdf(double p) {
            // quantile function (norminv):
            // https://jp.mathworks.com/help/stats/norminv.html
            return -1 * std::sqrt(2) * boost::math::erfc_inv(2 * p);
        }
        double truncated_normal(double a, double b) {
            // random sampling from truncated normal:
            // https://people.sc.fsu.edu/~jburkardt/cpp_src/truncated_normal/truncated_normal.html
            double a_cdf = _normal_cdf(a);
            double b_cdf = _normal_cdf(b);
            double u = uniform(a_cdf, b_cdf);
            if (u == 0.0) {
                u = MIN_VAL;
            } else if (u == 1.0) {
                u = MAX_VAL;
            }
            return _inverse_normal_cdf(u);
        }
        int binomial(int n, double p) {
            binomial_distribution<int> distribution(n, p);
            return distribution(mt);
        }
        int multinomial(size_t k, double* p) {
            // random sampling from multinomial
            // https://github.com/ampl/gsl/blob/master/randist/multinomial.c#L44-L78
            double norm = 0.0, sum_p = 0.0;
            for (int i=0; i<k; ++i) {
                norm += p[i];
            }
            for (int i=0; i<k; ++i) {
                if (p[i] > 0.0) {
                    int ret = binomial(1, p[i] / (norm - sum_p));
                    if (ret) return i;
                }
                sum_p += p[i];
            }
            return k - 1;
        }
    }
}