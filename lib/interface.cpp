#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

#include "interface.hpp"
#include "mis_interface.hpp"
#include "mcl_interface.hpp"


PYBIND11_MODULE(interface, m) {
    m.def("cvrp_two_opt", &cvrp_two_opt_interface, "");
    m.def("cvrp_random_two_opt", &cvrp_random_two_opt_interface, "");
    m.def("cvrp_greedy_insert", &cvrp_greedy_insert_interface, "");
    m.def("cvrp_partially_greedy_insert", &cvrp_partially_greedy_insert_interface, "");
    m.def("cvrp_neighbors2sol", &cvrp_neighbors2sol_interface, "");
    m.def("cvrp_init_insertion_state", &cvrp_init_insertion_state_interface, "");
    m.def("cvrp_eval_cost", &cvrp_eval_cost, "");

    m.def("tsp_two_opt_inplace", &tsp_two_opt_inplace_interface, "");
    m.def("tsp_random_two_opt_inplace", &tsp_random_two_opt_inplace_interface, "");
    m.def("tsp_double_two_opt", &tsp_double_two_opt_interface, "");
    m.def("tsp_greedy_insert", &tsp_greedy_insert_interface, "");
    m.def("tsp_partially_greedy_insert", &tsp_partially_greedy_insert_interface, "");
    m.def("tsp_neighbors2sol", tsp_neighbors2sol, "");
    m.def("tsp_init_insertion_state", &tsp_init_insertion_state_interface, "");
    m.def("tsp_eval_cost", &tsp_eval_cost, "");

    py::class_<std::shared_ptr<std::vector<std::vector<std::vector<int>>>>>(m, "MISBatchedNeighborsInt32");
    m.def("mis_edges2neighbors", &mis_edges2neighbors_interface_int32, "");
    m.def("mis_edges2neighbors_nocast", &mis_edges2neighbors_interface_int32_nocast, py::return_value_policy::take_ownership);
    m.def("mis_partially_greedy_insert", &mis_partially_greedy_insert_interface_int32, "");
    m.def("mis_init_insertion_state", &mis_init_insertion_state_interface_int32, "");
    m.def("mis_partially_mask", &mis_partially_mask_interface, "");

    m.def("mcl_partially_greedy_insert", &mcl_partially_greedy_insert_interface<int>, "");

    m.def("argsort_uint8", &argsort_interface_uint8, "");
    m.def("argsort_fp32", &argsort_interface_fp32, "");
}
