#pragma once

#include <algorithm>
#include <cmath>


inline void softmax_(float * x, const int n) {
    float max_val = *std::max_element(x, x + n);
    float t = 0.;
    for (int i = 0; i < n; ++i) {
        x[i] -= max_val;
        x[i] = std::exp(x[i]);
        t += x[i];
    }
    t = 1. / t;
    for (int i = 0; i < n; ++i) {
        x[i] *= t; 
    }
}


inline void softmax(float * out, const float * x, const int n) {
    float max_val = *std::max_element(x, x + n);
    float t = 0.;
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] - max_val;
        out[i] = std::exp(out[i]);
        t += out[i];
    }
    t = 1. / t;
    for (int i = 0; i < n; ++i) {
        out[i] *= t; 
    }
}
