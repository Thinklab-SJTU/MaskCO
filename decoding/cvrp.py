import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import threading
import typing as tp
import tqdm
from concurrent.futures import ThreadPoolExecutor
from models import CVRPModel, CVRPModelConfig
from helpers.coord_transform import normalize, cdist
from training import load_ckpt
from lib import cvrp_two_opt, CVRPPartialInsertion, cvrp_eval_cost
from decoding.utils import (
    convert_heatmap_dtype, 
    CoordDynamicAugment as DynamicAugment,
    compute_num_depot,
)


def cvrp_searching_decode(
    dataset: dict[str, np.ndarray],
    capacity: int,
    penalty: float,
    model: CVRPModel,
    sampling_steps: int,
    cycles: int,
    keep_rate: float,
    batch_size: int,
    runs: int,
    two_opt_steps: int,
    disable_gumbel: bool,
    gumbel_scale_factor: float,
    heatmap_dtype: jax.typing.DTypeLike,
    topk: int | None,
    augment_level: int,
    padding_policy: tp.Literal['none', 'auto'],
    threads_over_batches: int | None,
    seed: int,
):  
    num_workers = 1

    np.random.seed(seed)
    
    coords = dataset['coords']
    unnormalized_demands = dataset['demands']
    
    num_instances = coords.shape[0]
    num_nodes = coords.shape[1] - 1

    dist_mat_np = np.array(jax.jit(cdist)(coords))

    if 'opt_costs' in dataset.keys():
        opt_costs = dataset['opt_costs']
        mean_opt_cost = opt_costs.mean().item()
    elif 'routes' in dataset.keys():
        opt_routes = dataset['routes']
        opt_costs = cvrp_eval_cost(opt_routes, dist_mat_np, unnormalized_demands, capacity)
        mean_opt_cost = opt_costs.mean().item()
    else:
        opt_costs = None
        mean_opt_cost = None

    raw_features = np.concatenate([
        coords, unnormalized_demands[..., None] / capacity
    ], axis=-1).astype(np.float32)

    if padding_policy == 'none':
        expected_padded_length = None
    elif padding_policy == 'auto':
        expected_padded_length = compute_num_depot(
            total_demand = unnormalized_demands.sum(axis=-1).max().item(),
            capacity=capacity,
        ) + coords.shape[1] - 1
    else:
        raise ValueError()
    
    def _encode(raw_features: jax.Array):
        raw_features = raw_features.at[..., :2].set(normalize(raw_features[..., :2]))
        features = model.encode(raw_features)
        return features
    
    def _decode_step(
        features: jax.Array, 
        neighbors: jax.Array,
        gumbel_key: jax.Array | None = None,
    ):
        @jax.vmap
        def neighbors2adj(neighbors: jax.Array):
            neighbors = jnp.where(
                neighbors < 0,
                num_nodes * 2,  # out-of-bound values
                neighbors,
            )
            adjmat = jnp.zeros([num_nodes + 1, num_nodes + 1], dtype=jnp.int8)
            _arange = jnp.arange(neighbors.shape[0])
            adjmat = adjmat.at[_arange[1:], neighbors[1:, 0]].set(1, mode='drop')
            adjmat = adjmat.at[_arange[1:], neighbors[1:, 1]].add(1, mode='drop')
            adjmat = adjmat.at[0, :].set(adjmat[:, 0])
            return adjmat
        
        adjmat = neighbors2adj(neighbors)    

        timestep = (neighbors >= 0).sum(axis=[-1, -2]) / (2 * num_nodes)  # more accurate estimation
        def denoised_fn(adjmat: jax.Array, timestep: jax.Array):
            logits = model.decode(features, timestep, adjmat.astype(jnp.float16))
            if gumbel_key is not None:
                logits = logits + jax.random.gumbel(gumbel_key, logits.shape, logits.dtype) * gumbel_scale_factor
            adjmat = jax.nn.softmax(logits, axis=-1) * 2
            return adjmat
        adjmat_pred = denoised_fn(adjmat, timestep)
        adjmat_pred = jnp.where(adjmat, 0., adjmat_pred)
        adjmat_pred = adjmat_pred.at[:, 0].set(0)
        adjmat_pred = convert_heatmap_dtype(adjmat_pred, dtype=heatmap_dtype)
        if topk is None or topk <= 0:
            candidate_edges = jnp.argsort(
                adjmat_pred.reshape(batch_size, -1),
                axis=-1, descending=True, stable=False,
            )
        else:
            _, candidate_edges = jax.lax.top_k(
                adjmat_pred.reshape(batch_size, -1),
                k=topk,
            )
        candidate_edges = jnp.stack(jnp.divmod(candidate_edges, num_nodes + 1), axis=-1)
        return candidate_edges

    encode = jax.jit(_encode)
    decode_step = jax.jit(_decode_step)

    num_kept_edges = round(keep_rate * num_nodes)
    generation_node_times_schedule = np.linspace(0, num_nodes * 2, num=sampling_steps + 1, dtype=np.float32).astype(np.int32)
    searching_node_times_schedule = np.linspace(num_kept_edges * 2, num_nodes * 2, num=sampling_steps + 1, dtype=np.float32).astype(np.int32)
    
    def inference_fn(seed: int, raw_features: np.ndarray, dist_mat: np.ndarray, unnormalized_demands: np.ndarray):
        coords, normalized_demands = raw_features[..., :2], raw_features[..., 2:]
        # features: jax.Array = encode(raw_features)
        features_manager = DynamicAugment(
            lambda x: encode(np.concatenate([x, normalized_demands], axis=-1)),
            coords, 
            augment_level=augment_level,
        )
        storage: dict[int, np.ndarray] = {}
        def _single_run(r: int):
            generator = np.random.default_rng(seed + r * 77)
            insertion = CVRPPartialInsertion(batch_size, num_nodes + 1, num_workers=num_workers)
            # generation
            for i in range(sampling_steps):
                candidate_edges = decode_step(
                    features_manager(generator), 
                    insertion.neighbors, 
                    None if disable_gumbel else
                    generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
                )
                insertion.insert(np.array(candidate_edges), generation_node_times_schedule[i + 1])
            sols = insertion.get_sols()
            sols = cvrp_two_opt(
                sols, dist_mat, unnormalized_demands, 
                capacity, penalty, num_steps=two_opt_steps, 
                num_workers=num_workers,
            )
            # record cost if feasible
            costs = cvrp_eval_cost(sols, dist_mat, unnormalized_demands, capacity, num_workers=num_workers)
            # searching
            min_costs = costs
            for _ in range(cycles - 1):
                edges = sols * (num_nodes + 1) + np.roll(sols, shift=1, axis=-1)
                generator.permuted(edges, axis=1, out=edges)
                edges = np.stack(np.divmod(edges, num_nodes + 1), axis=-1)
                edges[:, num_kept_edges:] = num_nodes * 2   # out-of-bound value
                insertion.set_state(np.ascontiguousarray(edges[:, :num_kept_edges]))
                for i in range(sampling_steps):
                    candidate_edges = decode_step(
                        features_manager(generator), 
                        insertion.neighbors, 
                        None if disable_gumbel else
                        generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
                    )
                    insertion.insert(np.array(candidate_edges), searching_node_times_schedule[i + 1])
                sols = insertion.get_sols()
                # TODO: check whether additional dummy depots are needed here   
                if expected_padded_length is not None:
                    if expected_padded_length > sols.shape[-1]:
                        sols = np.ascontiguousarray(
                            np.pad(sols, [(0, 0), (0, expected_padded_length - sols.shape[-1])], mode='constant')
                        )
                sols = cvrp_two_opt(
                    sols, dist_mat, unnormalized_demands, 
                    capacity, penalty, num_steps=two_opt_steps, 
                    num_workers=num_workers,
                )
                # record cost if feasible
                costs = cvrp_eval_cost(sols, dist_mat, unnormalized_demands, capacity, num_workers=num_workers)
                min_costs = np.minimum(min_costs, costs)
            storage[r] = min_costs
        
        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        min_costs = storage[0]
        for i in range(1, runs):
            this_costs = storage[i]
            min_costs = np.minimum(min_costs, this_costs)
        return min_costs

    def run_batch(i: int) -> float:
        cost = inference_fn(
            i * 888,
            raw_features[i * batch_size:(i + 1) * batch_size], 
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
            unnormalized_demands[i * batch_size:(i + 1) * batch_size],
        )
        return cost.mean().item()
    
    # triton autotune and warmup
    for _ in range(5):
        i = 0
        generator = np.random.default_rng(i)
        features = encode(raw_features[i * batch_size:(i + 1) * batch_size])
        candidate_edges = decode_step(
            features, 
            jnp.full([batch_size, num_nodes + 1, 2], num_nodes * 2, dtype=jnp.int32),
            None if disable_gumbel else
            generator.integers(0, 1 << 31, size=[2], dtype=np.uint32),
        )
        del features, candidate_edges

    costs: list[float] = []
    if threads_over_batches is None or threads_over_batches == 1:
        for i in tqdm.tqdm(range(num_instances // batch_size)):
            batch_cost = run_batch(i)
            print(f'batch {i} cost mean: {batch_cost:.6f}')
            costs.append(batch_cost)
    else:
        assert num_instances % (batch_size * threads_over_batches) == 0
        with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
            for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
                batch_id_start = j * threads_over_batches
                batch_id_end = batch_id_start + threads_over_batches
                submit_costs = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
                for i, batch_cost in zip(range(batch_id_start, batch_id_end), submit_costs):
                    print(f'batch {i} cost mean: {batch_cost:.6f}')
                    costs.append(batch_cost)

    mean_cost = sum(costs) / len(costs)
    print(f'mean cost: {mean_cost:.6f}')
    if mean_opt_cost is not None:
        print(f'opt cost: {mean_opt_cost:.6f}')
        print(f'Gap: {(mean_cost - mean_opt_cost) / mean_opt_cost * 100:.6f} %')

    return mean_cost


if __name__ == '__main__': 
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--penalty', type=float, default=3.)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_gumbel', action='store_true', default=False)
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--augment_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--gumbel_scale_factor', type=float, required=True)
    parser.add_argument('--padding_policy', type=str, default='none')

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: CVRPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    dataset = dict(np.load(args.data))

    cvrp_searching_decode(
        dataset, args.capacity, args.penalty, model,
        sampling_steps=args.sampling_steps, 
        cycles=args.cycles, keep_rate=args.keep_rate,
        batch_size=args.batch_size, runs=args.runs,
        two_opt_steps=args.two_opt_steps,
        disable_gumbel=args.disable_gumbel,
        gumbel_scale_factor=args.gumbel_scale_factor,
        padding_policy=args.padding_policy,
        heatmap_dtype=args.heatmap_dtype,
        topk=args.topk,
        augment_level=args.augment_level,
        threads_over_batches=args.threads_over_batches,
        seed=args.seed,
    )
    
