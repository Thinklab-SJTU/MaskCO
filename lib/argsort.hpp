#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>


template<typename T>
inline auto argsort(
    const T * data, 
    const int n,
    const bool descending = false
) -> std::vector<int> {
    // 创建索引数组
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // 根据 descending 参数选择排序方向
    if (descending) {
        std::sort(indices.begin(), indices.end(),
            [&data](int i, int j) {
                return data[i] > data[j];
            }
        );
    } else {
        std::sort(indices.begin(), indices.end(),
            [&data](int i, int j) {
                return data[i] < data[j];
            }
        );
    }

    return indices;
}

// 针对 uint8_t 的基数排序实现
inline auto argsort_radix_uint8(
    const uint8_t * data, 
    const int n,
    const bool descending = false
) -> std::vector<int> {
    std::vector<int> output(n);
    std::array<size_t, 256> count = {0};

    // 统计每个值的出现次数
    for (int i = 0; i < n; ++i) {
        ++count[data[i]];
    }

    std::array<size_t, 256> sum;
    std::copy(count.begin(), count.end(), sum.begin());

    // 构造前缀和数组
    if (descending) {
        // 降序：从高位到低位累加
        for (int i = 254; i >= 0; --i) {
            sum[i] += sum[i + 1];
        }
    } else {
        // 升序：从低位到高位累加
        for (int i = 1; i < 256; ++i) {
            sum[i] += sum[i - 1];
        }
    }

    // 调整前缀和为索引位置
    for (auto& s : sum) {
        --s;
    }

    // 反向遍历原数组以保持稳定性
    for (int i = n - 1; i >= 0; --i) {
        uint8_t v = data[i];
        output[sum[v]] = i;
        sum[v]--;
    }

    return output;
}

// 模板特化：当 T 为 uint8_t 时，调用 argsort_radix_uint8
template<>
inline auto argsort<uint8_t>(
    const uint8_t * data, 
    const int n,
    const bool descending
) -> std::vector<int> {
    return argsort_radix_uint8(data, n, descending);
}


template<typename T>
inline void argsort(
    int * indices,
    const T * data, 
    const int n,
    const bool descending = false
) {
    // 创建索引数组
    for (int i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // 根据 descending 参数选择排序方向
    if (descending) {
        std::sort(indices, indices + n,
            [&data](int i, int j) {
                return data[i] > data[j];
            }
        );
    } else {
        std::sort(indices, indices + n,
            [&data](int i, int j) {
                return data[i] < data[j];
            }
        );
    }
}


template<>
inline void argsort<uint8_t>(
    int * indices,
    const uint8_t * data, 
    const int n,
    const bool descending
) {
    std::array<size_t, 256> count = {0};

    // 统计每个值的出现次数
    for (int i = 0; i < n; ++i) {
        ++count[data[i]];
    }

    std::array<size_t, 256> sum;
    std::copy(count.begin(), count.end(), sum.begin());

    // 构造前缀和数组
    if (descending) {
        // 降序：从高位到低位累加
        for (int i = 254; i >= 0; --i) {
            sum[i] += sum[i + 1];
        }
    } else {
        // 升序：从低位到高位累加
        for (int i = 1; i < 256; ++i) {
            sum[i] += sum[i - 1];
        }
    }

    // 调整前缀和为索引位置
    for (auto& s : sum) {
        --s;
    }

    // 反向遍历原数组以保持稳定性
    for (int i = n - 1; i >= 0; --i) {
        uint8_t v = data[i];
        indices[sum[v]] = i;
        sum[v]--;
    }
}
