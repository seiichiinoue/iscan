#pragma once
#include <iostream>
#include <cmath>
#include "fmath.hpp"
#define NUM_SENSE 8
#define NUM_TIME 10
#define KAPPA_PHI 4.0
#define KAPPA_PSI 10.0
#define GAMMA_A 7.0
#define GAMMA_B 3.0
#define CONTEXT_WINDOW_WIDTH 10
#define START_YEAR 1810
#define END_YEAR 2010
#define YEAR_INTERVAL 20
#define BURN_IN_PERIOD 500
#define IGNORE_WORD_COUNT 3
#define KAPPA_PHI_START 100
#define KAPPA_PHI_INTERVAL 50
#define MIN_VAL 1e-9
#define MAX_VAL 1.0 - MIN_VAL

namespace scan {
    double logsumexp(double x, double y, bool flg) {
        if (flg) return y; // init mode
        if (x == y) return x + std::log(2);
        double vmin = std::min(x, y);
        double vmax = std::max(x, y);
        if (vmax > vmin + 50) {
            return vmax;
        } else {
            return vmax + std::log(std::exp(vmin - vmax) + 1.0);
        }
    }
}