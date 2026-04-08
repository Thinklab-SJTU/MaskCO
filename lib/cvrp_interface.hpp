#pragma once

#include "pybind11/cast.h"
#include "pybind11/gil.h"
#include "pybind11/pytypes.h"
#include "cvrp.hpp"
#include "par_utils.hpp"
#include "tsp_interface.hpp"

#include <algorithm>
#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sys/types.h>
#include <vector>

namespace py = pybind11;


inline py::array_t<int> cvrp_two_opt_interface(
    const py::array_t<int> & routes, const py::array_t<float> & dist_mat, 
    const py::array_t<int> & demands, const int capacity, const float penalty, const int num_steps, const int num_workers
) {
    assert(routes.flags() & py::array::c_style);
    assert(dist_mat.flags() & py::array::c_style);
    assert(demands.flags() & py::array::c_style);

    const bool batched = routes.ndim() == 2;

    if (!batched) {
        // write back to pyarray fmt
        auto new_routes = py::array_t<int>(routes.size());
        auto new_routes_ptr = static_cast<int *>(new_routes.request().ptr);
        const int num_nodes = dist_mat.shape()[1];

        pybind11::gil_scoped_release release;

        std::vector<std::vector<int>> routes_cpp_fmt;
        // find first zero and read
        int first_zero_pos;
        for (first_zero_pos = 0; first_zero_pos < routes.size(); ++first_zero_pos) {
            if (*routes.data(first_zero_pos) == 0) {
                break;
            }
        }
        assert(first_zero_pos < routes.size());

        for (int i = 0; i < routes.size(); ++i) {
            auto & idx = *routes.data((i + first_zero_pos) % routes.size());
            if (idx == 0) {
                routes_cpp_fmt.emplace_back(std::vector<int>());
            }
            (routes_cpp_fmt.end() - 1)->push_back(idx);
        }

        cvrp::two_opt(routes_cpp_fmt, dist_mat.data(), demands.data(), num_nodes, capacity, penalty, num_steps);

        int i = 0;
        for (const auto & route: routes_cpp_fmt) {
            for (const auto & idx: route) {
                new_routes_ptr[i] = idx;
                ++i;
            }
        }

        return new_routes;
    } else {
        const int batch_size = routes.shape()[0];
        const int num_nodes = dist_mat.shape()[1];
        const int num_nodes_with_replicated_depots = routes.shape()[1];

        auto new_routes = py::array_t<int>({batch_size, num_nodes_with_replicated_depots});
        auto new_routes_ptr_base = static_cast<int *>(new_routes.request().ptr);

        pybind11::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            const auto routes_ptr = routes.data() + task_id * num_nodes_with_replicated_depots;
            const auto dist_mat_ptr = dist_mat.data() + task_id * num_nodes * num_nodes;
            const auto demands_ptr = demands.data() + task_id * num_nodes;

            std::vector<std::vector<int>> routes_cpp_fmt;
            // find first zero and read
            int first_zero_pos;
            for (first_zero_pos = 0; first_zero_pos < num_nodes_with_replicated_depots; ++first_zero_pos) {
                if (routes_ptr[first_zero_pos] == 0) {
                    break;
                }
            }
            assert(first_zero_pos < num_nodes_with_replicated_depots);

            for (int i = 0; i < num_nodes_with_replicated_depots; ++i) {
                auto & idx = routes_ptr[(i + first_zero_pos) % num_nodes_with_replicated_depots];
                if (idx == 0) {
                    routes_cpp_fmt.emplace_back(std::vector<int>());
                }
                (routes_cpp_fmt.end() - 1)->push_back(idx);
            }

            cvrp::two_opt(routes_cpp_fmt, dist_mat_ptr, demands_ptr, num_nodes, capacity, penalty, num_steps);

            auto new_routes_ptr = new_routes_ptr_base + task_id * num_nodes_with_replicated_depots;

            int i = 0;
            for (const auto & route: routes_cpp_fmt) {
                for (const auto & idx: route) {
                    new_routes_ptr[i] = idx;
                    ++i;
                }
            }
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        return new_routes;
    }
}


inline py::array_t<int> cvrp_random_two_opt_interface(const int seed, const py::array_t<int> & routes, const int num_steps) {
    assert(routes.flags() & py::array::c_style);

    const int routes_size = static_cast<int>(routes.size());
    auto new_routes = py::array_t<int>(routes_size);
    auto new_routes_ptr = static_cast<int *>(new_routes.request().ptr);
    auto routes_ptr = static_cast<int *>(routes.request().ptr);

    pybind11::gil_scoped_release release;

    std::vector<std::vector<int>> routes_cpp_fmt;
    // find first zero and read
    int first_zero_pos;
    for (first_zero_pos = 0; first_zero_pos < routes_size; ++first_zero_pos) {
        if (routes_ptr[first_zero_pos] == 0) {
            break;
        }
    }
    assert(first_zero_pos < routes_size);

    for (int i = 0; i < routes_size; ++i) {
        auto & idx = routes_ptr[(i + first_zero_pos) % routes_size];
        if (idx == 0) {
            routes_cpp_fmt.emplace_back(std::vector<int>());
        }
        (routes_cpp_fmt.end() - 1)->push_back(idx);
    }

    cvrp::random_two_opt(seed, routes_cpp_fmt, num_steps);

    // write back to pyarray fmt
    int i = 0;
    for (const auto & route: routes_cpp_fmt) {
        for (const auto & idx: route) {
            new_routes_ptr[i] = idx;
            ++i;
        }
    }

    return new_routes;
} 


inline auto
cvrp_greedy_insert_interface(
    const py::array_t<int> & candidate_edges, const int num_nodes, const int num_workers
) {


    assert(candidate_edges.flags() & py::array::c_style);

    const bool batched = candidate_edges.ndim() == 3;
    auto candidate_edges_ptr = candidate_edges.data();
    
    if (!batched) {
        const int num_candidate_edges = candidate_edges.shape()[0];
        pybind11::gil_scoped_release release;
        auto route_cpp_fmt = cvrp::greedy_insert(candidate_edges_ptr, num_nodes, num_candidate_edges);
        int route_len = 0;
        for (const auto & subroute: route_cpp_fmt) {
            route_len += static_cast<int>(subroute.size());
        }
        pybind11::gil_scoped_acquire acquire;
        py::array_t<int> route(route_len);
        auto route_ptr = static_cast<int *>(route.request().ptr);
        cvrp::cppfmt2arrayfmt(route_ptr, route_cpp_fmt);
        return route;
    } else {
        const int batch_size = candidate_edges.shape()[0];
        const int num_candidate_edges = candidate_edges.shape()[1];

        pybind11::gil_scoped_release release;

        std::vector<std::vector<std::vector<int>>> batched_routes_cpp_fmt(batch_size);
        std::vector<int> batched_route_len(batch_size);

        auto task_fn = [&](const int task_id) {
            auto routes_cpp_fmt = cvrp::greedy_insert(
                candidate_edges_ptr + num_candidate_edges * 2 * task_id, 
                num_nodes, num_candidate_edges
            );
            batched_routes_cpp_fmt[task_id] = routes_cpp_fmt;
            int route_len = 0;
            for (const auto & subroute: routes_cpp_fmt) {
                route_len += static_cast<int>(subroute.size());
            }
            batched_route_len[task_id] = route_len;
        };
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
        const int max_route_len = *std::max_element(batched_route_len.begin(), batched_route_len.end());

        pybind11::gil_scoped_acquire acquire;
        auto batched_routes = py::array_t<int>({batch_size, max_route_len});
        auto batched_routes_ptr = static_cast<int *>(batched_routes.request().ptr);
        pybind11::gil_scoped_release release_2;

        std::fill(batched_routes_ptr, batched_routes_ptr + batch_size * max_route_len, 0);
        for (int i = 0; i < batch_size; ++i) {
            cvrp::cppfmt2arrayfmt(batched_routes_ptr + i * max_route_len, batched_routes_cpp_fmt[i]);
        }

        return batched_routes;
    }
}


inline auto
cvrp_partially_greedy_insert_interface(
    py::array_t<int> & subroute_id,
    py::array_t<int> & neighbors,
    py::array_t<int> & num_inserted_node_times,
    const py::array_t<int> & candidate_edges, 
    const int target_node_times,
    const int num_workers
) {


    assert(subroute_id.flags() & py::array::c_style);
    assert(neighbors.flags() & py::array::c_style);
    assert(num_inserted_node_times.flags() & py::array::c_style);
    assert(candidate_edges.flags() & py::array::c_style);

    const bool batched = subroute_id.ndim() == 2;

    auto candidate_edges_ptr = candidate_edges.data();
    auto subroute_id_ptr = static_cast<int *>(subroute_id.request().ptr);
    auto neighbors_ptr = static_cast<int *>(neighbors.request().ptr);
    auto num_inserted_node_times_ptr = static_cast<int *>(num_inserted_node_times.request().ptr);

    if (!batched) {
        const int num_nodes = subroute_id.shape()[0];
        const int num_candidate_edges = candidate_edges.shape()[0];
        
        py::gil_scoped_release release;
        
        const int num_inserted_node_times_this_round = cvrp::partially_greedy_insert(
            subroute_id_ptr, 
            neighbors_ptr, 
            candidate_edges_ptr, 
            num_nodes,
            num_candidate_edges,
            *num_inserted_node_times_ptr, 
            target_node_times - (*num_inserted_node_times_ptr)
        );
        (*num_inserted_node_times_ptr) += num_inserted_node_times_this_round;
    } else {
        const int batch_size = subroute_id.shape()[0];
        const int num_nodes = subroute_id.shape()[1];
        const int num_candidate_edges = candidate_edges.shape()[1];

        py::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            const int num_inserted_node_times_this_round = cvrp::partially_greedy_insert(
                subroute_id_ptr + task_id * num_nodes, 
                neighbors_ptr + task_id * num_nodes * 2, 
                candidate_edges_ptr + task_id * num_candidate_edges * 2, 
                num_nodes,
                num_candidate_edges,
                num_inserted_node_times_ptr[task_id], 
                target_node_times - num_inserted_node_times_ptr[task_id]
            );
            num_inserted_node_times_ptr[task_id] += num_inserted_node_times_this_round;
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
    }

}


inline auto
cvrp_neighbors2sol_interface(
    const py::array_t<int> & neighbors,
    const int num_workers
) {
    assert(neighbors.flags() & py::array::c_style);

    const bool batched = neighbors.ndim() == 3;
    auto neighbors_ptr = neighbors.data();

    if (!batched) {
        const int num_nodes = neighbors.shape()[0];

        py::gil_scoped_release release;

        auto route_cpp_fmt = cvrp::neighbors2sol(neighbors_ptr, num_nodes);
        int route_len = 0;
        for (const auto & subroute: route_cpp_fmt) {
            route_len += static_cast<int>(subroute.size());
        }

        pybind11::gil_scoped_acquire acquire;

        py::array_t<int> route(route_len);
        auto route_ptr = static_cast<int *>(route.request().ptr);

        pybind11::gil_scoped_release release_2;

        cvrp::cppfmt2arrayfmt(route_ptr, route_cpp_fmt);
        return route;
    } else {
        const int batch_size = neighbors.shape()[0];
        const int num_nodes = neighbors.shape()[1];

        py::gil_scoped_release release;

        std::vector<std::vector<std::vector<int>>> batched_routes_cpp_fmt(batch_size);
        std::vector<int> batched_route_len(batch_size);
        auto task_fn = [&](const int task_id) {
            auto routes_cpp_fmt = cvrp::neighbors2sol(
                neighbors_ptr + task_id * num_nodes * 2, 
                num_nodes
            );
            batched_routes_cpp_fmt[task_id] = routes_cpp_fmt;
            int route_len = 0;
            for (const auto & subroute: routes_cpp_fmt) {
                route_len += static_cast<int>(subroute.size());
            }
            batched_route_len[task_id] = route_len;
        };
        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
        const int max_route_len = *std::max_element(batched_route_len.begin(), batched_route_len.end());

        pybind11::gil_scoped_acquire acquire;

        auto batched_routes = py::array_t<int>({batch_size, max_route_len});
        auto batched_routes_ptr = static_cast<int *>(batched_routes.request().ptr);

        pybind11::gil_scoped_release release_2;

        std::fill(batched_routes_ptr, batched_routes_ptr + batch_size * max_route_len, 0);
        for (int i = 0; i < batch_size; ++i) {
            cvrp::cppfmt2arrayfmt(batched_routes_ptr + i * max_route_len, batched_routes_cpp_fmt[i]);
        }

        return batched_routes;
    }
}


inline auto
cvrp_init_insertion_state_interface(
    py::array_t<int> & neighbors, py::array_t<int> & subroute_id, 
    py::array_t<int> & num_inserted_node_times,
    const py::array_t<int> & edges,
    const int num_workers
) {
    assert(neighbors.flags() & py::array::c_style);
    assert(subroute_id.flags() & py::array::c_style);
    assert(num_inserted_node_times.flags() & py::array::c_style);
    assert(edges.flags() & py::array::c_style);

    const bool batched = neighbors.ndim() == 3;

    auto edges_ptr = edges.data();
    auto neighbors_ptr = static_cast<int *>(neighbors.request().ptr);
    auto subroute_id_ptr = static_cast<int *>(subroute_id.request().ptr);
    auto num_inserted_node_times_ptr = static_cast<int *>(num_inserted_node_times.request().ptr);

    if (!batched) {
        const int num_nodes = neighbors.shape()[0];
        const int num_edges = edges.shape()[0]; 

        py::gil_scoped_release release;

        cvrp::init_insertion_state(
            neighbors_ptr, 
            subroute_id_ptr, 
            num_inserted_node_times_ptr[0], 
            edges_ptr, 
            num_nodes, 
            num_edges
        );
    } else {
        const int batch_size = neighbors.shape()[0];
        const int num_nodes = neighbors.shape()[1];
        const int num_edges = edges.shape()[1];
        
        py::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            cvrp::init_insertion_state(
                neighbors_ptr + task_id * num_nodes * 2, 
                subroute_id_ptr + task_id * num_nodes, 
                num_inserted_node_times_ptr[task_id], 
                edges_ptr + task_id * num_edges * 2, 
                num_nodes, 
                num_edges
            );
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));
    }
}


inline auto
cvrp_eval_cost(
    const py::array_t<int> & route,
    const py::array_t<float> & dist_mat,
    const py::array_t<int> & demands,
    const int capacity,
    const int num_workers
) {
    assert(route.flags() & py::array::c_style);
    assert(dist_mat.flags() & py::array::c_style);
    assert(demands.flags() & py::array::c_style);

    const bool batched = route.ndim() == 2;
    auto route_ptr = route.data();
    auto dist_mat_ptr = dist_mat.data();
    auto demands_ptr = demands.data();

    if (!batched) {
        const int num_nodes = dist_mat.shape()[0];
        const int route_len = route.shape()[0];

        auto cost = py::array_t<float>({});
        auto cost_ptr = static_cast<float *>(cost.request().ptr); 

        py::gil_scoped_release release;

        cvrp::eval_cost(*cost_ptr, route_ptr, dist_mat_ptr, demands_ptr, capacity, num_nodes, route_len);

        return cost;
    } else {
        const int batch_size = dist_mat.shape()[0];
        const int num_nodes = dist_mat.shape()[1];
        const int route_len = route.shape()[1];

        auto cost = py::array_t<float>(batch_size);
        auto cost_ptr = static_cast<float *>(cost.request().ptr); 

        py::gil_scoped_release release;

        auto task_fn = [&](const int task_id) {
            cvrp::eval_cost(
                cost_ptr[task_id], 
                route_ptr + task_id * route_len, 
                dist_mat_ptr + task_id * num_nodes * num_nodes, 
                demands_ptr + task_id * num_nodes, 
                capacity, 
                num_nodes, 
                route_len
            );
        };

        parallelize(task_fn, batch_size, std::min(batch_size, num_workers));

        return cost;
    }
}
