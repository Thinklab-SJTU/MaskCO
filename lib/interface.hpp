#pragma once

#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "cvrp.hpp"
#include "par_utils.hpp"
#include "tsp.hpp"
#include "argsort.hpp"
#include "tsp_interface.hpp"
#include "cvrp_interface.hpp"
#include "mis_interface.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace py = pybind11; 


template<typename T>
inline py::array_t<int>
argsort_interface(
    const py::array_t<T> & data, const bool descending = false, int num_workers = 1
) {
    const bool batched = data.ndim() == 2;

    auto data_ptr = data.data();

    if (!batched) {
        const int num_nodes = data.shape()[0];
        py::array_t<int> indices(num_nodes);
        auto indices_ptr = static_cast<int *>(indices.request().ptr);

        pybind11::gil_scoped_release release;

        argsort(indices_ptr, data_ptr, num_nodes, descending);
        
        return indices;
    } else {
        const int batch_size = data.shape()[0];
        const int num_nodes = data.shape()[1];
        py::array_t<int> indices({batch_size, num_nodes});
        auto indices_ptr = static_cast<int *>(indices.request().ptr);

        pybind11::gil_scoped_release release;

        num_workers = std::min(num_workers, batch_size);
        auto task_fn = [&](const int task_id) {
            argsort(indices_ptr + task_id * num_nodes, data_ptr + task_id * num_nodes, num_nodes, descending);
        };

        parallelize(task_fn, batch_size, num_workers);

        return indices;
    }
}


inline py::array_t<int>
argsort_interface_uint8(
    const py::array_t<uint8_t> & data, const bool descending, int num_workers
) {
    return argsort_interface(data, descending, num_workers);
}


inline py::array_t<int>
argsort_interface_fp32(
    const py::array_t<float> & data, const bool descending, int num_workers
) {
    return argsort_interface(data, descending, num_workers);
}
