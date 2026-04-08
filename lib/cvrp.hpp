#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <tuple>
#include <random>

#include "params.hpp"


namespace route {
// depot is always 0 and route[0] is always depot

inline std::tuple<int, int, float>      // {i, j, cost_reduction}
find_best_intra_two_opt(const std::vector<int> & route, const float * dist_mat, const int num_nodes) {
    const int route_len = route.size();
    
    float best_cost_reduction = std::numeric_limits<float>().min();
    int best_i = 0, best_j = 0;
    for (int i = 0; i < route_len - 2; ++i) {
        for (int j = i + 2; j < route_len; ++j) {
            const float cost_reduction = (
                dist_mat[route[i] * num_nodes + route[i + 1]] + dist_mat[route[j] * num_nodes + route[(j + 1) % route_len]]
                - dist_mat[route[i] * num_nodes + route[j]] - dist_mat[route[i + 1] * num_nodes + route[(j + 1) % route_len]]
            );
            if constexpr (TWO_OPT_TYPE == TwoOptType::Best) {
                if (cost_reduction > best_cost_reduction) {
                    best_cost_reduction = cost_reduction;
                    best_i = i;
                    best_j = j;
                }
            } else {
                if (cost_reduction > 0.) {
                    return {i, j, cost_reduction};
                }
            }
        }
    }

    return {best_i, best_j, best_cost_reduction};
}


template<bool star = false>
inline std::tuple<int, int, float, int, int>      // {i, j, penalized_cost_reduction, new_route_u_capacity, new_route_v_capacity}
find_best_inter_two_opt(
    const std::vector<int> & route_u, const std::vector<int> & route_v,
    const int route_u_capacity, const int route_v_capacity,
    const float * dist_mat, const int * demands,
    const int num_nodes, const int capacity, const float penalty
) {
    assert(route_u[0] == 0 && route_v[0] == 0);

    const int route_u_len = route_u.size();
    const int route_v_len = route_v.size();

    float best_penalized_cost_reduction = std::numeric_limits<float>().min();
    int best_i = 0, best_j = 0;
    int best_cum_u_capacity = 0, best_cum_v_capacity = 0;

    int cum_u_capacity = 0;
    for (int i = 0; i < route_u_len; ++i) {
        auto & x = route_u[i];
        auto & y = route_u[(i + 1) % route_u_len];
        cum_u_capacity += demands[x];

        int cum_v_capacity = 0;
        for (int j = 0; j < route_v_len; ++j) {
            // if ((i == 0 || i == route_u_len - 1) && (j == 0 || j == route_v_len - 1)) {
            //     break;
            // }       // TODO: check necessity

            auto & s = route_v[j];
            auto & t = route_v[(j + 1) % route_v_len];
            cum_v_capacity += demands[s];

            float cost_reduction;
            if constexpr (star) {
                cost_reduction = (
                    dist_mat[x * num_nodes + y] + dist_mat[s * num_nodes + t]
                    - dist_mat[x * num_nodes + t] - dist_mat[s * num_nodes + y]
                );
            } else {
                cost_reduction = (
                    dist_mat[x * num_nodes + y] + dist_mat[t * num_nodes + s]
                    - dist_mat[x * num_nodes + s] - dist_mat[t * num_nodes + y]
                );
            }
            float capacity_exceed_reduction;
            if constexpr (star) {
                capacity_exceed_reduction = static_cast<float>((
                    std::max(0, route_u_capacity - capacity) + std::max(0, route_v_capacity - capacity)
                ) - (
                    std::max(0, cum_u_capacity + (route_v_capacity - cum_v_capacity) - capacity)
                    + std::max(0, cum_v_capacity + (route_u_capacity - cum_u_capacity) - capacity)
                ));
            } else {
                capacity_exceed_reduction = static_cast<float>((
                    std::max(0, route_u_capacity - capacity) + std::max(0, route_v_capacity - capacity)
                ) - (
                    std::max(0, cum_u_capacity + cum_v_capacity - capacity)
                    + std::max(0, (route_v_capacity - cum_v_capacity) + (route_u_capacity - cum_u_capacity) - capacity)
                ));
            }
            const float penalized_cost_reduction = cost_reduction + capacity_exceed_reduction * penalty;

            if constexpr (TWO_OPT_TYPE == TwoOptType::Best) {
                if (penalized_cost_reduction > best_penalized_cost_reduction) {
                    best_penalized_cost_reduction = penalized_cost_reduction;
                    best_i = i;
                    best_j = j;
                    best_cum_u_capacity = cum_u_capacity;
                    best_cum_v_capacity = cum_v_capacity;
                }
            } else {
                if (penalized_cost_reduction > 0.) {
                    best_penalized_cost_reduction = penalized_cost_reduction;
                    best_i = i;
                    best_j = j;
                    best_cum_u_capacity = cum_u_capacity;
                    best_cum_v_capacity = cum_v_capacity;

                    int new_route_u_capacity, new_route_v_capacity;
                    if constexpr (star) {
                        new_route_u_capacity = best_cum_u_capacity + (route_v_capacity - best_cum_v_capacity);
                        new_route_v_capacity = best_cum_v_capacity + (route_u_capacity - best_cum_u_capacity);
                    } else {
                        new_route_u_capacity = best_cum_u_capacity + best_cum_v_capacity;
                        new_route_v_capacity = (route_v_capacity - best_cum_v_capacity) + (route_u_capacity - best_cum_u_capacity);
                    }
                    return {best_i, best_j, best_penalized_cost_reduction, new_route_u_capacity, new_route_v_capacity};
                }
            }
        }
    }

    int new_route_u_capacity, new_route_v_capacity;
    if constexpr (star) {
        new_route_u_capacity = best_cum_u_capacity + (route_v_capacity - best_cum_v_capacity);
        new_route_v_capacity = best_cum_v_capacity + (route_u_capacity - best_cum_u_capacity);
    } else {
        new_route_u_capacity = best_cum_u_capacity + best_cum_v_capacity;
        new_route_v_capacity = (route_v_capacity - best_cum_v_capacity) + (route_u_capacity - best_cum_u_capacity);
    }

    return {best_i, best_j, best_penalized_cost_reduction, new_route_u_capacity, new_route_v_capacity};
}


inline void apply_intra_two_opt(std::vector<int> & route, const int i, const int j) {
    std::reverse(route.begin() + i + 1, route.begin() + j + 1);
}


template<bool star = false>  // true: (x, y), (s, t) -> (x, t), (s, y); false: (x, y), (s, t) -> (x, s), (t, y)
inline void apply_inter_two_opt(std::vector<int> & route_u, std::vector<int> & route_v, const int i, const int j) {
    assert(route_u[0] == 0 && route_v[0] == 0);

    if constexpr (star) {
        const int u_to_exchange_len = static_cast<int>(route_u.size()) - i - 1; 
        const int v_to_exchange_len = static_cast<int>(route_v.size()) - j - 1;
        if (u_to_exchange_len >= v_to_exchange_len) {
            for (int d = 1; d <= v_to_exchange_len; ++d) {
                std::swap(route_u[i + d], route_v[j + d]);
            }
            for (int d = v_to_exchange_len + 1; d <= u_to_exchange_len; ++d) {
                route_v.push_back(route_u[i + d]);
            }
            route_u.erase(route_u.begin() + i + v_to_exchange_len + 1, route_u.end());
        } else {
            for (int d = 1; d <= u_to_exchange_len; ++d) {
                std::swap(route_v[j + d], route_u[i + d]);
            }
            for (int d = u_to_exchange_len + 1; d <= v_to_exchange_len; ++d) {
                route_u.push_back(route_v[j + d]);
            }
            route_v.erase(route_v.begin() + j + u_to_exchange_len + 1, route_v.end());
        }
    } else {
        // TODO: need further optimization
        std::reverse(route_v.begin() + 1, route_v.end());
        apply_inter_two_opt<true>(route_u, route_v, i, static_cast<int>(route_v.size()) - j - 1);
    }
}


inline int compute_route_capacity(const std::vector<int> & route, const int * demands) {
    int result = 0;
    for (const auto & node: route) {
        result += demands[node];
    }
    return result;
}

}   // namespace route


namespace cvrp {


inline void 
cppfmt2arrayfmt(
    int * route, 
    const std::vector<std::vector<int>> route_cpp_fmt
) {
    int idx = 0;
    for (const auto & subroute: route_cpp_fmt) {
        for (const auto & node_idx: subroute) {
            route[idx] = node_idx;
            ++idx;
        } 
    }
}


inline std::vector<std::vector<int>>
arrayfmt2cppfmt(
    const int * route, const int route_len
) {
    assert(route[0] == 0);
    std::vector<std::vector<int>> route_cpp_fmt;

    for (int i = 0; i < route_len; ++i) {
        if (route[i] == 0) {
            route_cpp_fmt.emplace_back(std::vector<int>());
        }
        (route_cpp_fmt.end() - 1)->emplace_back(route[i]);
    }

    return route_cpp_fmt;
}


inline void two_opt(
    std::vector<std::vector<int>> & routes, const float * dist_mat, const int * demands, 
    const int num_nodes, const int capacity, const float penalty, const int num_steps
) {
    if (num_steps <= 0) {
        return;
    }

    std::vector<int> route_capacity;
    route_capacity.reserve(routes.size());
    for (const auto & route: routes) {
        route_capacity.push_back(route::compute_route_capacity(route, demands));
    }

    for (int step_ctr = 0; step_ctr < num_steps; ++step_ctr) {
        float penalized_cost_reduction = std::numeric_limits<float>().min();
        int i, j, new_route_u_capacity, new_route_v_capacity;
        int route_u_idx, route_v_idx;
        int move_type = 0;  // 0: intra, 1: inter, 2: inter *

        // intra 2-opt
        for (int u = 0; u < static_cast<int>(routes.size()); ++u) {
            auto result = route::find_best_intra_two_opt(routes[u], dist_mat, num_nodes);
            if (std::get<2>(result) > penalized_cost_reduction) {
                std::tie(i, j, penalized_cost_reduction) = result;
                route_u_idx = u;
            }
        }

        for (int u = 0; u < static_cast<int>(routes.size()) - 1; ++u) {
            for (int v = u + 1; v < static_cast<int>(routes.size()); ++v) {
                // inter 2-opt
                auto result = route::find_best_inter_two_opt<false>(
                    routes[u], routes[v], route_capacity[u], route_capacity[v], 
                    dist_mat, demands, num_nodes, capacity, penalty
                );
                if (std::get<2>(result) > penalized_cost_reduction) {
                    std::tie(i, j, penalized_cost_reduction, new_route_u_capacity, new_route_v_capacity) = result;
                    route_u_idx = u;
                    route_v_idx = v;
                    move_type = 1;
                }
                // inter 2-opt *
                result = route::find_best_inter_two_opt<true>(
                    routes[u], routes[v], route_capacity[u], route_capacity[v], 
                    dist_mat, demands, num_nodes, capacity, penalty
                );
                if (std::get<2>(result) > penalized_cost_reduction) {
                    std::tie(i, j, penalized_cost_reduction, new_route_u_capacity, new_route_v_capacity) = result;
                    route_u_idx = u;
                    route_v_idx = v;
                    move_type = 2;
                }
            }
        }

        if (penalized_cost_reduction <= 1e-5) {     // to further investigate
            break;
        }   // search completed

        if (move_type == 0) {
            route::apply_intra_two_opt(routes[route_u_idx], i, j);
        } else if (move_type == 1) {
            route::apply_inter_two_opt<false>(routes[route_u_idx], routes[route_v_idx], i, j);
            route_capacity[route_u_idx] = new_route_u_capacity;
            route_capacity[route_v_idx] = new_route_v_capacity;
        } else {    // move_type == 2
            route::apply_inter_two_opt<true>(routes[route_u_idx], routes[route_v_idx], i, j);
            route_capacity[route_u_idx] = new_route_u_capacity;
            route_capacity[route_v_idx] = new_route_v_capacity;
        }
    }
}


inline void random_two_opt(std::mt19937 & generator, std::vector<std::vector<int>> & routes, const int num_steps) {
    // uniformly sample a random 2-opt(*) and apply
    int num_routes_len_sum = 0;
    int num_intra_two_opt = 0;
    for (const auto & route: routes) {
        const auto route_len = static_cast<int>(route.size());  // cast to int to enable negative value 
        num_routes_len_sum += route_len;
        num_intra_two_opt += std::max(0, route_len * (route_len - 3) / 2);
    }
    int num_inter_two_opt = 0;  // shared by inter 2-opt and inter 2-opt*
    for (int u = 0; u < routes.size() - 1; ++u) {
        for (int v = u + 1; v < routes.size(); ++v) {
            num_inter_two_opt += static_cast<int>(routes[u].size() * routes[v].size());
        }
    }
    for (int step_ctr = 0; step_ctr < num_steps; ++step_ctr) {
        int move = std::uniform_int_distribution<int>(0, num_inter_two_opt + 2 * num_intra_two_opt - 1)(generator);
        if (move < num_intra_two_opt) {    // intra 2-opt
            for (auto & route: routes) {
                const auto route_len = static_cast<int>(route.size());  // cast to int to enable negative value 
                const auto num_moves_this_route = std::max(0, route_len * (route_len - 3) / 2);
                if (move < num_moves_this_route) {
                    // transform move to i, j need more computation than resample i, j, so we resample
                    const int i = std::uniform_int_distribution<int>(0, route_len - 1)(generator);
                    // route_len > 3 has been already satisfied 
                    const int diff = std::uniform_int_distribution<int>(0, route_len - 3 - 1)(generator);   
                    const int j = (i + diff + 2) % route_len;
                    if (i <= j) {
                        route::apply_intra_two_opt(route, i, j);
                    } else {
                        route::apply_intra_two_opt(route, j, i);
                    }
                    goto ThisStepCompleted;
                } else {
                    move -= num_moves_this_route;
                }
            }
            throw std::logic_error("Unreachable code.");
        } else {     // inter 2-opt(*)
            const bool star = (move >= num_intra_two_opt + num_inter_two_opt);
            if (star) {
                move -= num_intra_two_opt + num_inter_two_opt;
            } else {
                move -= num_intra_two_opt;
            }
            for (int u = 0; u < routes.size() - 1; ++u) {
                for (int v = u + 1; v < routes.size(); ++v) {
                    const auto num_moves_this_route_pair = static_cast<int>(routes[u].size() * routes[v].size());
                    if (move < num_moves_this_route_pair) {
                        const int i = move / static_cast<int>(routes[v].size());
                        const int j = move % static_cast<int>(routes[v].size());

                        // as such move changes route_u_len and route_v_len, we need update num_intra_two_opt and num_inter_two_opt
                        // intra
                        int route_len_u = static_cast<int>(routes[u].size());
                        num_intra_two_opt -= std::max(0, route_len_u * (route_len_u - 3) / 2);
                        int route_len_v = static_cast<int>(routes[v].size());
                        num_intra_two_opt -= std::max(0, route_len_v * (route_len_v - 3) / 2);
                        // inter
                        num_inter_two_opt -= (route_len_u + route_len_v) * (num_routes_len_sum - (route_len_u + route_len_v)) + route_len_u * route_len_v;
                        if (star) {
                            route::apply_inter_two_opt<true>(routes[u], routes[v], i, j);
                        } else {
                            route::apply_inter_two_opt<false>(routes[u], routes[v], i, j);
                        }
                        // intra
                        route_len_u = static_cast<int>(routes[u].size());
                        num_intra_two_opt += std::max(0, route_len_u * (route_len_u - 3) / 2);
                        route_len_v = static_cast<int>(routes[v].size());
                        num_intra_two_opt += std::max(0, route_len_v * (route_len_v - 3) / 2);
                        // inter
                        num_inter_two_opt += (route_len_u + route_len_v) * (num_routes_len_sum - (route_len_u + route_len_v)) + route_len_u * route_len_v;
                        goto ThisStepCompleted;
                    } else {
                        move -= num_moves_this_route_pair;
                    }
                }
            }
        } 
        ThisStepCompleted: ;
    }
}


inline void random_two_opt(const unsigned long seed, std::vector<std::vector<int>> & routes, const int num_steps) {
    auto generator = std::mt19937(seed);
    cvrp::random_two_opt(generator, routes, num_steps);
}


inline std::vector<int> get_pos2route(const int * routes, const int num_nodes) {
    assert(routes[0] == 0);

    std::vector<int> pos2route;
    pos2route.resize(num_nodes);
    int route_idx = -1;
    for (int i = 0; i < num_nodes; ++i) {
        route_idx += static_cast<int>(routes[i] == 0);
        pos2route[i] = route_idx;
    }
    return pos2route;
}


inline auto     // we may also need stop condition provided by probabilities
greedy_insert(
    const int * candidate_edges, 
    const int num_nodes,
    const int num_candidate_edges
) {

    std::vector<int> subroute_id;
    subroute_id.resize(num_nodes);
    std::fill(subroute_id.begin(), subroute_id.end(), 0);     // `0` means has not been inserted
                                                         // the subroute_id of depot `subroute_id[0]` is always `0`

    auto update_id = [&subroute_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subroute_id[i] == from) {
                subroute_id[i] = to;
            }
        }
    };

    std::vector<int> neighbors;
    neighbors.resize(2 * num_nodes);
    std::fill(neighbors.begin(), neighbors.end(), -1);  // `-1` means undetermined
                                                        // the neighbors of depot are always `-1`
    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (i != 0) {
            if (neighbors[2 * i] == -1) {
                neighbors[2 * i] = j;
            } else {
                assert(neighbors[2 * i + 1] == -1);
                neighbors[2 * i + 1] = j;
            }
        }
        if (j != 0) {
            if (neighbors[2 * j] == -1) {
                neighbors[2 * j] = i;
            } else {
                assert(neighbors[2 * j + 1] == -1);
                neighbors[2 * j + 1] = i;
            }
        }
    };

    int next_available_subroute = 1;
    int num_inserted_node_times = 0;    // should be `2 * (num_nodes - 1)` after insertion
    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int i = candidate_edges[2 * edge_idx];
        const int j = candidate_edges[2 * edge_idx + 1];

        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        if (i == j) {
            continue;
        }

        const int including_depot = static_cast<int>(i == 0 || j == 0);

        if (subroute_id[i] == 0) {
            num_inserted_node_times += 2 - including_depot;
            if (subroute_id[j] == 0) {
                if (i != 0) {
                    subroute_id[i] = next_available_subroute;
                }
                if (j != 0) {
                    subroute_id[j] = next_available_subroute;
                }
                ++next_available_subroute;
            } else {
                if (i != 0) {
                    subroute_id[i] = subroute_id[j];
                }
            }
            set_neighbor(i, j);
        } else {
            if (subroute_id[j] == 0) {
                num_inserted_node_times += 2 - including_depot;
                if (j != 0) {
                    subroute_id[j] = subroute_id[i];
                }
                set_neighbor(i, j);
            } else {
                // both have been inserted
                if (subroute_id[i] == subroute_id[j]) {
                    // same subtour, cannot insert
                    continue;
                } else {
                    num_inserted_node_times += 2 - including_depot;
                    update_id(subroute_id[j], subroute_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        if (num_inserted_node_times == 2 * (num_nodes - 1)) {
            break;
        }
    }
    assert(num_inserted_node_times == 2 * (num_nodes - 1));

    // neighbors -> route
    int num_lefted_nodes = num_nodes - 1;
    bool has_visited[num_nodes];    // `has_visited[0]` has a special meaning
    std::fill(has_visited, has_visited + num_nodes, false);
    has_visited[0] = true;
    std::vector<std::vector<int>> route;
    while (true) {
        std::vector<int> subroute;
        subroute.emplace_back(0);
        int cur_node = 1;
        for (; cur_node < num_nodes; ++cur_node) {
            if (!has_visited[cur_node]) {
                if (neighbors[2 * cur_node] == 0 || neighbors[2 * cur_node + 1] == 0) {
                    break;
                }
            }
        }   // set to the first unvisited node with a depot neighbor
        while (true) {
            subroute.emplace_back(cur_node);
            --num_lefted_nodes;
            has_visited[cur_node] = true;
            const int & neighbor_1 = neighbors[2 * cur_node];
            const int & neighbor_2 = neighbors[2 * cur_node + 1];
            if (!has_visited[neighbor_1]) {
                cur_node = neighbor_1;
                continue;
            } else {
                if (!has_visited[neighbor_2]) {
                    cur_node = neighbor_2;
                    continue;
                } else {
                    assert(neighbor_1 == 0 || neighbor_2 == 0);
                    break;
                }
            }
        }
        route.emplace_back(subroute);
        if (num_lefted_nodes <= 0) {
            break;
        }
    }
    assert(num_lefted_nodes == 0);
    return route;
}


inline int
fill_solution_naive(
    int * subroute_id,
    int * neighbors,
    const int num_nodes,
    const int num_inserted_node_times,
    const int num_node_times_to_insert     
);


inline auto
partially_greedy_insert(
    int * subroute_id,
    int * neighbors,
    const int * candidate_edges, 
    const int num_nodes,
    const int num_candidate_edges,
    const int num_inserted_node_times,
    const int num_node_times_to_insert
) {


    auto update_id = [&subroute_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subroute_id[i] == from) {
                subroute_id[i] = to;
            }
        }
    };
    
    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (i != 0) {
            if (neighbors[2 * i] == -1) {
                neighbors[2 * i] = j;
            } else {
                assert(neighbors[2 * i + 1] == -1);
                neighbors[2 * i + 1] = j;
            }
        }
        if (j != 0) {
            if (neighbors[2 * j] == -1) {
                neighbors[2 * j] = i;
            } else {
                assert(neighbors[2 * j + 1] == -1);
                neighbors[2 * j + 1] = i;
            }
        }
    };

    int next_available_subroute = num_inserted_node_times + 1;
    int num_inserted_node_times_this_round = 0;

    for (int edge_idx = 0; edge_idx < num_candidate_edges; ++edge_idx) {
        const int i = candidate_edges[2 * edge_idx];
        const int j = candidate_edges[2 * edge_idx + 1];

        if (neighbors[2 * i + 1] != -1 || neighbors[2 * j + 1] != -1) {
            continue;
        }
        if (i == j) {
            continue;
        }

        const int including_depot = static_cast<int>(i == 0 || j == 0);

        if (subroute_id[i] == 0) {
            num_inserted_node_times_this_round += 2 - including_depot;
            if (subroute_id[j] == 0) {
                if (i != 0) {
                    subroute_id[i] = next_available_subroute;
                }
                if (j != 0) {
                    subroute_id[j] = next_available_subroute;
                }
                ++next_available_subroute;
            } else {
                if (i != 0) {
                    subroute_id[i] = subroute_id[j];
                }
            }
            set_neighbor(i, j);
        } else {
            if (subroute_id[j] == 0) {
                num_inserted_node_times_this_round += 2 - including_depot;
                if (j != 0) {
                    subroute_id[j] = subroute_id[i];
                }
                set_neighbor(i, j);
            } else {
                // both have been inserted
                if (subroute_id[i] == subroute_id[j]) {
                    // same subtour, cannot insert
                    continue;
                } else {
                    num_inserted_node_times_this_round += 2 - including_depot;
                    update_id(subroute_id[j], subroute_id[i]);
                    set_neighbor(i, j);
                }
            }
        }

        if (num_inserted_node_times + num_inserted_node_times_this_round == 2 * (num_nodes - 1)) {
            break;
        }
        if (num_inserted_node_times_this_round >= num_node_times_to_insert) {
            break;
        }
    }

    if constexpr (ENABLE_INSERTION_FALLBACK) {
        if (
            !( num_inserted_node_times + num_inserted_node_times_this_round == 2 * (num_nodes - 1) 
            || num_inserted_node_times_this_round >= num_node_times_to_insert)
        ) {
            const int num_filled_node_times = fill_solution_naive(
                subroute_id, neighbors, num_nodes, 
                num_inserted_node_times + num_inserted_node_times_this_round, 
                num_node_times_to_insert - num_inserted_node_times_this_round
            );
            return num_filled_node_times + num_inserted_node_times_this_round;
        }
    } else {
        assert(
            num_inserted_node_times + num_inserted_node_times_this_round == 2 * (num_nodes - 1) 
            || num_inserted_node_times_this_round >= num_node_times_to_insert
        );  // `num_inserted_node_times_this_round > num_node_times_to_insert` may hold since every insertion can increase 
            // `num_inserted_node_times_this_round' 1 or 2
    }
    
    return num_inserted_node_times_this_round;
}


inline int
fill_solution_naive(
    int * subroute_id,
    int * neighbors,
    const int num_nodes,
    const int num_inserted_node_times,
    const int num_node_times_to_insert     
) {
    std::vector<int> free_nodes;
    for (int i = 1; i < num_nodes; ++i) {
        if (neighbors[2 * i] == -1 || neighbors[2 * i + 1] == -1) {
            free_nodes.emplace_back(i);
        }
    }
    free_nodes.emplace_back(0);     // depot

    const int num_free_nodes = static_cast<int>(free_nodes.size());
    std::vector<int> candidate_edges;
    candidate_edges.reserve(num_free_nodes * (num_free_nodes - 1));

    for (int idx_1 = 0; idx_1 < num_free_nodes - 1; ++idx_1) {
        const int & node_x = free_nodes[idx_1];
        for (int idx_2 = idx_1 + 1; idx_2 < num_free_nodes; ++idx_2) {
            const int & node_y = free_nodes[idx_2];
            candidate_edges.emplace_back(node_x);
            candidate_edges.emplace_back(node_y);
        }
    }

    return partially_greedy_insert(
        subroute_id, neighbors, 
        candidate_edges.data(), num_nodes, 
        candidate_edges.size() / 2, 
        num_inserted_node_times, 
        num_node_times_to_insert
    );
}


inline auto
neighbors2sol(
    const int * neighbors,
    const int num_nodes
) {
    // neighbors -> route
    int num_lefted_nodes = num_nodes - 1;
    bool has_visited[num_nodes];    // `has_visited[0]` has a special meaning
    std::fill(has_visited, has_visited + num_nodes, false);
    has_visited[0] = true;
    std::vector<std::vector<int>> route;
    while (true) {
        std::vector<int> subroute;
        subroute.emplace_back(0);
        int cur_node = 1;
        for (; cur_node < num_nodes; ++cur_node) {
            if (!has_visited[cur_node]) {
                if (neighbors[2 * cur_node] == 0 || neighbors[2 * cur_node + 1] == 0) {
                    break;
                }
            }
        }   // set to the first unvisited node with a depot neighbor
        while (true) {
            subroute.emplace_back(cur_node);
            --num_lefted_nodes;
            has_visited[cur_node] = true;
            const int & neighbor_1 = neighbors[2 * cur_node];
            const int & neighbor_2 = neighbors[2 * cur_node + 1];
            if (!has_visited[neighbor_1]) {
                cur_node = neighbor_1;
                continue;
            } else {
                if (!has_visited[neighbor_2]) {
                    cur_node = neighbor_2;
                    continue;
                } else {
                    assert(neighbor_1 == 0 || neighbor_2 == 0);
                    break;
                }
            }
        }
        route.emplace_back(subroute);
        if (num_lefted_nodes <= 0) {
            break;
        }
    }
    assert(num_lefted_nodes == 0);
    return route;
}


inline void
init_insertion_state(
    int * neighbors, int * subroute_id, 
    int & num_inserted_node_times,
    const int * edges,
    const int num_nodes, const int num_edges
) {


    auto update_id = [&subroute_id, &num_nodes](const int from, const int to) {
        for (int i = 0; i < num_nodes; ++i) {
            if (subroute_id[i] == from) {
                subroute_id[i] = to;
            }
        }
    };
    
    auto set_neighbor = [&neighbors](const int & i, const int & j) {
        if (i != 0) {
            if (neighbors[2 * i] == -1) {
                neighbors[2 * i] = j;
            } else {
                assert(neighbors[2 * i + 1] == -1);
                neighbors[2 * i + 1] = j;
            }
        }
        if (j != 0) {
            if (neighbors[2 * j] == -1) {
                neighbors[2 * j] = i;
            } else {
                assert(neighbors[2 * j + 1] == -1);
                neighbors[2 * j + 1] = i;
            }
        }
    };

    num_inserted_node_times = 0;
    std::fill(subroute_id, subroute_id + num_nodes, 0);
    std::fill(neighbors, neighbors + 2 * num_nodes, -1);
    int next_available_subroute = 1;

    for (int edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
        const int i = edges[2 * edge_idx];
        const int j = edges[2 * edge_idx + 1];

        const int including_depot = static_cast<int>(i == 0 || j == 0);
        if (i == j) {
            continue;
        }   // It is necessary because `[0, 0]` can occur!

        if (subroute_id[i] == 0) {
            num_inserted_node_times += 2 - including_depot;
            if (subroute_id[j] == 0) {
                if (i != 0) {
                    subroute_id[i] = next_available_subroute;
                }
                if (j != 0) {
                    subroute_id[j] = next_available_subroute;
                }
                ++next_available_subroute;
            } else {
                if (i != 0) {
                    subroute_id[i] = subroute_id[j];
                }
            }
            set_neighbor(i, j);
        } else {
            if (subroute_id[j] == 0) {
                num_inserted_node_times += 2 - including_depot;
                if (j != 0) {
                    subroute_id[j] = subroute_id[i];
                }
                set_neighbor(i, j);
            } else {
                if (subroute_id[i] == subroute_id[j]) {
                    assert(false);
                } else {
                    num_inserted_node_times += 2 - including_depot;
                    update_id(subroute_id[j], subroute_id[i]);
                    set_neighbor(i, j);
                }
            }
        }
    }
}


inline void
eval_cost(
    float & cost,
    const int * route,
    const float * dist_mat,
    const int * demands,
    const int capacity,
    const int num_nodes,
    const int route_len
) {
    assert(route[0] == 0);

    cost = 0.;
    int cum_demand = 0;
    for (int pos = 0; pos < route_len; ++pos) {
        const int & i = route[pos];
        const int & j = route[(pos + 1) % route_len]; 
        cost += dist_mat[i * num_nodes + j];
        
        cum_demand += demands[i];
        if (j == 0) {
            if (cum_demand > capacity) {
                cost = std::numeric_limits<float>().max() / 2;
                return;
            }
            cum_demand = 0;       
        }
    }
}


}   // namespace cvrp
