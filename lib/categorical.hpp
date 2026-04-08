#pragma once

#include <random>


inline int categorical(std::mt19937 & generator, const float * probs, const int num_classes, const float probs_sum = 1.) {
    std::uniform_real_distribution<float> dist(0., probs_sum);

    float threshold = dist(generator);

    for (int i = 0; i < num_classes - 1; ++i) {
        threshold -= probs[i];
        if (threshold < 0.) {
            return i;
        }
    }
    return num_classes - 1;
}
