import numpy as np
import typing as tp

from .interface import (
    cvrp_two_opt as _cvrp_two_opt_impl,
    tsp_two_opt_inplace as _tsp_two_opt_inplace_impl,
    tsp_random_two_opt_inplace as _tsp_random_two_opt_inplace_impl,
    tsp_double_two_opt as _tsp_double_two_opt_impl,
    tsp_greedy_insert as _tsp_greedy_insert_impl,
    tsp_partially_greedy_insert as _tsp_partially_greedy_insert_impl,
    tsp_init_insertion_state as _tsp_init_insertion_state_impl,
    tsp_neighbors2sol as _tsp_neighbors2sol_impl,
    tsp_eval_cost as _tsp_eval_cost_impl,
    argsort_uint8 as _argsort_uint8_impl,
    argsort_fp32 as _argsort_fp32_impl,
    cvrp_random_two_opt as _cvrp_random_two_opt_impl,
    cvrp_greedy_insert as _cvrp_greedy_insert_impl,
    cvrp_partially_greedy_insert as _cvrp_partially_greedy_insert_impl,
    cvrp_neighbors2sol as _cvrp_neighbors2sol_impl,
    cvrp_init_insertion_state as _cvrp_init_insertion_state_impl,
    cvrp_eval_cost as _cvrp_eval_cost_impl,
    mis_edges2neighbors as _mis_edges2neighbors_impl,
    mis_edges2neighbors_nocast as _mis_edges2neighbors_nocast_impl,
    mis_partially_greedy_insert as _mis_partially_greedy_insert_impl,
    mis_init_insertion_state as _mis_init_insertion_state_impl,
    mis_partially_mask as _mis_partially_mask_impl,
    mcl_partially_greedy_insert as _mcl_partially_greedy_insert_impl,
)

MISBatchedNeighbors = tp.Any


def tsp_eval_cost(
    tour: np.ndarray,
    dist_mat: np.ndarray,
    num_workers: int,
) -> np.ndarray:
    return _tsp_eval_cost_impl(
        tour, dist_mat, num_workers,
    )


def cvrp_eval_cost(
    route: np.ndarray,
    dist_mat: np.ndarray,
    demands: np.ndarray,
    capacity: int,
    num_workers: int = 1
) -> np.ndarray:
    ''' Evaluate the cost of a cvrp solution. Return a large positive value if infeasible.
    '''

    return _cvrp_eval_cost_impl(
        route, dist_mat,
        demands, capacity,
        num_workers,
    )


def cvrp_two_opt(
    routes: np.ndarray, dist_mat: np.ndarray, demands: np.ndarray, 
    capacity: int, penalty: float, num_steps: int, num_workers: int = 1,
) -> np.ndarray:
    return _cvrp_two_opt_impl(routes, dist_mat, demands, capacity, penalty, num_steps, num_workers)


def cvrp_random_two_opt(seed: int, route: np.ndarray, num_steps: int) -> np.ndarray:
    assert route.ndim == 1
    return _cvrp_random_two_opt_impl(seed, route, num_steps)


def cvrp_greedy_insert(
    candidate_edges: np.ndarray, num_nodes: int, num_workers: int = 1,
    *, safe: bool = True,
) -> np.ndarray:
    '''
    :param num_nodes: Number of total nodes, including the depot.
    '''

    if safe:
        edges_to_ensure_safe = np.stack([np.zeros(num_nodes - 1, dtype=np.int32), np.arange(1, num_nodes, dtype=np.int32)], axis=-1)
        if candidate_edges.ndim == 2:
            candidate_edges = np.concatenate([candidate_edges, edges_to_ensure_safe], axis=0)
        else:
            assert candidate_edges.ndim == 3
            edges_to_ensure_safe = np.stack([edges_to_ensure_safe] * candidate_edges.shape[0], axis=0)
            candidate_edges = np.concatenate([candidate_edges, edges_to_ensure_safe], axis=1)
    return _cvrp_greedy_insert_impl(candidate_edges, num_nodes, num_workers)


def tsp_two_opt(
    tours: np.ndarray, dist_mat: np.ndarray, num_steps: int, num_workers: int = 1, inplace: bool = False,
) -> np.ndarray:
    if not inplace:
        tours = tours.copy()
    _tsp_two_opt_inplace_impl(tours, dist_mat, num_steps, num_workers)
    return tours


def tsp_random_two_opt(
    seed: int, tours: np.ndarray, num_steps: np.ndarray, num_workers: int = 1, inplace: bool = False,
) -> np.ndarray:
    if not inplace:
        tours = tours.copy()
    _tsp_random_two_opt_inplace_impl(seed, tours, num_steps, num_workers)
    return tours


def tsp_double_two_opt(
    seed: int, tours: np.ndarray, dist_mat: np.ndarray,
    steps: int, random_steps: np.ndarray, 
    num_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    return _tsp_double_two_opt_impl(seed, tours, dist_mat, steps, random_steps, num_workers)


def tsp_greedy_insert(
    candidate_edges: np.ndarray, num_nodes: int, num_workers: int = 1,
) -> np.ndarray:
    return _tsp_greedy_insert_impl(candidate_edges, num_nodes, num_workers)


def argsort(data: np.ndarray, descending: bool = False, num_workers: int = 1) -> np.ndarray:
    dtype = np.dtype(data.dtype)
    if dtype == 'uint8':
        return _argsort_uint8_impl(data, descending, num_workers)
    elif dtype == 'float32':
        return _argsort_fp32_impl(data, descending, num_workers)
    else:
        raise ValueError()
    

def tsp_neighbors2sol(neighbors: np.ndarray, num_workers: int = 1) -> np.ndarray:
    return _tsp_neighbors2sol_impl(neighbors, num_workers)
    

def tsp_partially_greedy_insert(
    edges: np.ndarray, 
    subtour_id: np.ndarray,
    neighbors: np.ndarray,
    candidate_edges: np.ndarray, 
    num_inserted_edges: int,
    num_edges_to_insert: int,
    num_workers: int = 1, 
):
    _tsp_partially_greedy_insert_impl(
        edges, subtour_id,
        neighbors, candidate_edges,
        num_inserted_edges, num_edges_to_insert,
        num_workers,
    )


def tsp_init_insertion_state(
    neighbors: np.ndarray, subtour_id: np.ndarray,
    edges: np.ndarray, num_workers: int = 1,
):
    _tsp_init_insertion_state_impl(
        neighbors, subtour_id,
        edges, num_workers,
    )


class TSPPartialInsertion:
    def __init__(
        self,
        batch_size: int,
        num_nodes: int,
        num_workers: int = 1,
    ):
        self.subtour_id = np.zeros([batch_size, num_nodes], dtype=np.int32)
        self.neighbors = np.full([batch_size, num_nodes, 2], fill_value=-1, dtype=np.int32)
        self.num_inserted_edges = 0

        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_workers = num_workers
    
    def reset(self):
        self.subtour_id[:] = 0
        self.neighbors[:] = -1
        self.num_inserted_edges = 0

    def set_state(self, edges: np.ndarray):
        '''
        :param edges: with shape `[batch_size, num_edges, 2]`.
        '''
        tsp_init_insertion_state(self.neighbors, self.subtour_id, edges, self.num_workers)
        self.num_inserted_edges = edges.shape[1]

    def init_edges(self):
        return np.full([self.batch_size, self.num_nodes, 2], self.num_nodes + 1, dtype=np.int32)

    def insert(
        self,
        edges: np.ndarray,
        candidate_edges: np.ndarray,
        num_edges_to_insert: int
    ):
        tsp_partially_greedy_insert(
            edges, self.subtour_id,
            self.neighbors, candidate_edges,
            self.num_inserted_edges, num_edges_to_insert,
            self.num_workers,
        )
        self.num_inserted_edges += num_edges_to_insert

    def get_sols(self):
        assert self.num_inserted_edges == self.num_nodes
        return tsp_neighbors2sol(self.neighbors, self.num_workers)
    


def cvrp_partially_greedy_insert( 
    subroute_id: np.ndarray,
    neighbors: np.ndarray,
    num_inserted_node_times: np.ndarray,
    candidate_edges: np.ndarray, 
    target_node_times: int,
    num_workers: int = 1,
    *,
    safe: bool = True,
):
    ''' NEVER operate partial insertion on the SAME heatmap for more than ONE times! \
        Because edges like `[0, x]` can be inserted multiples time, which is not expected.
    :param target_node_times: Insertion will continue until `num_inserted_node_times >= target_node_times` or the route is complete. 
    '''

    if safe:
        num_nodes = subroute_id.shape[-1]
        edges_to_ensure_safe = np.stack([np.zeros(num_nodes - 1, dtype=np.int32), np.arange(1, num_nodes, dtype=np.int32)], axis=-1)
        if candidate_edges.ndim == 2:
            candidate_edges = np.concatenate([candidate_edges, edges_to_ensure_safe], axis=0)
        else:
            assert candidate_edges.ndim == 3
            edges_to_ensure_safe = np.stack([edges_to_ensure_safe] * candidate_edges.shape[0], axis=0)
            candidate_edges = np.concatenate([candidate_edges, edges_to_ensure_safe], axis=1)
    _cvrp_partially_greedy_insert_impl(
        subroute_id,
        neighbors, num_inserted_node_times,
        candidate_edges,
        target_node_times,
        num_workers,
    )


def cvrp_neighbors2sol(
    neighbors: np.ndarray,
    num_workers: int = 1,
) -> np.ndarray:
    return _cvrp_neighbors2sol_impl(
        neighbors,
        num_workers,
    )


def cvrp_init_insertion_state(
    subroute_id: np.ndarray,
    neighbors: np.ndarray,
    num_inserted_node_times: np.ndarray,
    edges: np.ndarray,
    num_workers: int = 1,
):
    _cvrp_init_insertion_state_impl(
        neighbors, subroute_id,
        num_inserted_node_times, edges,
        num_workers,
    )


class CVRPPartialInsertion:
    def __init__(
        self,
        batch_size: int,
        num_nodes: int,
        num_workers: int = 1,
    ):
        '''
        :param num_nodes: Number of total nodes, including the depot.
        '''
        
        self.subroute_id = np.zeros([batch_size, num_nodes], dtype=np.int32)
        self.neighbors = np.full([batch_size, num_nodes, 2], -1, dtype=np.int32)
        self.num_inserted_node_times = np.zeros([batch_size], dtype=np.int32)

        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.num_workers = num_workers

    def insert(
        self,
        candidate_edges: np.ndarray,
        target_node_times: int,
        safe: bool = True,
    ):
        cvrp_partially_greedy_insert(
            self.subroute_id, self.neighbors,
            self.num_inserted_node_times, candidate_edges,
            target_node_times, self.num_workers,
            safe=safe,
        )

    def reset(self):
        self.subroute_id[:] = 0
        self.neighbors[:] = -1
        self.num_inserted_node_times[:] = 0

    def get_sols(self):
        assert (self.num_inserted_node_times == 2 * (self.num_nodes - 1)).all()
        return cvrp_neighbors2sol(self.neighbors, self.num_workers)
    
    def set_state(self, edges: np.ndarray):
        cvrp_init_insertion_state(
            self.subroute_id, self.neighbors,
            self.num_inserted_node_times, edges,
            self.num_workers,
        )


def mis_edges2neighbors(
    edges: np.ndarray,
    num_nodes: np.ndarray,
    num_edges: np.ndarray,
    num_worker: int = 1,
    *,
    tolist: bool = True,
) -> list[list[list[int]]] | MISBatchedNeighbors:
    if tolist:
        return _mis_edges2neighbors_impl(
            edges, num_nodes, num_edges, num_worker,
        )
    else:
        return _mis_edges2neighbors_nocast_impl(
            edges, num_nodes, num_edges, num_worker,
        )


def mis_partially_greedy_insert(
    solution: np.ndarray,
    mask: np.ndarray,
    num_inserted_nodes: np.ndarray,
    num_masked_nodes: np.ndarray,
    candidate_nodes: np.ndarray,
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    target_num_nodes: np.ndarray,
    num_workers: int = 1,
):
    _mis_partially_greedy_insert_impl(
        solution, mask,
        num_inserted_nodes, num_masked_nodes,
        candidate_nodes, neighbors,
        num_nodes, target_num_nodes,
        num_workers,
    )


def mis_init_insertion_state(
    mask: np.ndarray,
    num_masked_nodes: np.ndarray,
    solution: np.ndarray,
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    num_workers: int = 1,
):
    _mis_init_insertion_state_impl(
        mask, num_masked_nodes,
        solution, neighbors,
        num_nodes, num_workers,
    )


def mis_partially_mask(
    solution: np.ndarray,
    seeds: np.ndarray,
    num_nodes: np.ndarray,
    num_nodes_to_keep: np.ndarray,
    num_workers: np.ndarray 
):
    assert seeds.dtype == np.int32

    _mis_partially_mask_impl(
        solution, seeds,
        num_nodes, num_nodes_to_keep,
        num_workers,
    )


class MISPartialInsertion:
    def __init__(
        self,
        neighbors: MISBatchedNeighbors,
        num_nodes: np.ndarray,
        num_nodes_padded: int,
        seed: int = 0,
        num_workers: int = 1,
    ):
        self.neighbors = neighbors
        self.num_nodes = num_nodes
        self.num_workers = num_workers

        batch_size = num_nodes.shape[0]

        self.solution = np.zeros([batch_size, num_nodes_padded], dtype='bool')
        self.mask = np.zeros([batch_size, num_nodes_padded], dtype='bool')
        self.num_inserted_nodes = np.zeros([batch_size], dtype=np.int32)
        self.num_masked_nodes = np.zeros([batch_size], dtype=np.int32)

        self.generator = np.random.default_rng(seed)
        self.batch_size = batch_size

    def insert(
        self,
        candidate_nodes: np.ndarray,
        target_num_nodes: np.ndarray,
    ):
        mis_partially_greedy_insert(
            self.solution, self.mask,
            self.num_inserted_nodes, self.num_masked_nodes,
            candidate_nodes, self.neighbors, self.num_nodes,
            target_num_nodes, self.num_workers,
        )

    def reset(self):
        self.solution[:] = False
        self.mask[:] = False
        self.num_inserted_nodes[:] = 0
        self.num_masked_nodes[:] = 0

    def set_state(self, solution: np.ndarray):
        self.solution = solution
        self.num_inserted_nodes = solution.sum(axis=-1, dtype=np.int32)
        mis_init_insertion_state(
            self.mask, self.num_masked_nodes,
            solution, self.neighbors,
            self.num_nodes, self.num_workers,
        )
    
    def partially_mask(self, keep_rate: float):
        num_nodes_to_keep = (keep_rate * self.num_inserted_nodes.astype(np.float32)).astype(np.int32)
        seeds = self.generator.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=[self.batch_size], dtype=np.int32)

        mis_partially_mask(
            self.solution, seeds,
            self.num_nodes, num_nodes_to_keep,
            self.num_workers,
        )
        self.set_state(self.solution)


def mis_greedy_insert(
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    num_nodes_padded: int,
    candidate_nodes: np.ndarray,
    num_workers: int = 1,
):
    insertion = MISPartialInsertion(
        neighbors,
        num_nodes,
        num_nodes_padded,
        num_workers=num_workers,
    )
    batch_size = num_nodes.shape[0]
    insertion.insert(
        candidate_nodes,
        target_num_nodes=np.full([batch_size], -1, dtype=np.int32),
    )
    return insertion.solution, insertion.num_inserted_nodes


def mcl_greedy_insert(
    neighbors: MISBatchedNeighbors,
    num_nodes: np.ndarray,
    num_nodes_padded: int,
    candidate_nodes: np.ndarray,
    num_workers: int = 1,
):
    batch_size = candidate_nodes.shape[0]
    
    solution = np.zeros([batch_size, num_nodes_padded], dtype='bool')
    num_clique_neighbors = np.zeros([batch_size, num_nodes_padded], dtype=np.int32)
    num_inserted_nodes = np.zeros([batch_size], dtype=np.int32)
    
    _mcl_partially_greedy_insert_impl(
        solution,
        num_clique_neighbors,
        num_inserted_nodes,
        candidate_nodes,
        neighbors,
        num_nodes,
        np.full([batch_size], fill_value=-1, dtype=np.int32),
        num_workers,
    )
    return solution, num_inserted_nodes


