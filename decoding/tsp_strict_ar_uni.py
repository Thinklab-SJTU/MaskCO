import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import threading
import tqdm
from concurrent.futures import ThreadPoolExecutor
from models import TSPModel, TSPModelConfig
from helpers.coord_transform import (
    normalize,
    cdist,
)
from helpers.tsp_transform import get_costs
from helpers import edges2adj
from training import load_ckpt
from lib import tsp_two_opt, TSPPartialInsertion, tsp_eval_cost
from decoding.utils import (
    convert_heatmap_dtype, 
    CoordDynamicAugment as DynamicAugment,
)
from functools import partial


def tsp_searching_decode(
    dataset: dict[str, np.ndarray],
    model: TSPModel, model_config: TSPModelConfig,
    sampling_steps: int,
    cycles: int,
    keep_rate: float,
    batch_size: int,
    runs: int,
    two_opt_steps: int,
    heatmap_dtype: jax.typing.DTypeLike,
    topk: int | None,
    augment_level: int,
    threads_over_batches: int | None,
    seed: int,
):  
    assert cycles == 1
    # assert runs == 1

    np.random.seed(seed)

    num_workers = 1

    assert {'coords'} <= set(dataset.keys()) <= {'coords', 'opt_tours', 'opt_costs'}

    coords = dataset['coords']
    dist_mat_np = np.array(jax.jit(cdist)(coords))

    if 'opt_costs' in dataset.keys():
        opt_costs = dataset['opt_costs']
        mean_opt_cost = opt_costs.mean().item()
    elif 'opt_tours' in dataset.keys():
        opt_tours = dataset['opt_tours']
        opt_costs = jax.jit(get_costs, backend='cpu')(opt_tours, dist_mat_np)
        mean_opt_cost = opt_costs.mean().item()
    else:
        opt_costs = None
        mean_opt_cost = None

    num_instances, num_nodes, _ = coords.shape
    num_nodes: int

    assert sampling_steps == num_nodes

    def _encode(raw_features: jax.Array):
        features = normalize(raw_features, centering_method='mean')
        features = model.encode(features)
        return features

    def _decode_step(
        features: jax.Array, 
        edges: jax.Array,
        start_node: jax.Array,
    ):
        timestep = (edges < num_nodes).sum(axis=[-1, -2]).astype(jnp.float32) / (2 * num_nodes)
        adjmat = edges2adj(edges, dtype=jnp.int8)
        logits = model.decode(features, timestep, adjmat.astype(jnp.float16))
        adjmat_pred = jax.nn.softmax(logits, axis=-1) * 2
        adjmat_pred = jnp.where(adjmat, 0., adjmat_pred)
        @jax.vmap
        def _mask_cannot_extend(adjmat_pred: jax.Array, adjmat: jax.Array, start_node: jax.Array):
            can_extend = adjmat.sum(-1) == 1

            is_first_step = ~can_extend.any()
            can_extend = can_extend.at[start_node].set(is_first_step)

            degrees, node_to_extend = jax.lax.top_k(can_extend, k=2)
            node_to_extend = node_to_extend.at[1].set(
                jnp.where(
                    degrees[1] == 0,
                    node_to_extend[0],
                    node_to_extend[1],
                )
            )

            adjmat_pred = jnp.where(
                jnp.logical_or(jnp.arange(num_nodes)[:, None] == node_to_extend[0], jnp.arange(num_nodes)[:, None] == node_to_extend[1]),
                adjmat_pred,
                0.,
            )
            return adjmat_pred
        adjmat_pred = _mask_cannot_extend(adjmat_pred, adjmat, start_node)

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

        candidate_edges = jnp.stack(jnp.divmod(candidate_edges, num_nodes), axis=-1)
        return candidate_edges
    
    encode = jax.jit(_encode)
    decode_step = jax.jit(_decode_step)
    get_costs_cpu = partial(tsp_eval_cost, num_workers=num_workers)

    num_kept_edges = round(keep_rate * num_nodes)
    generation_edges_schedule = np.linspace(0, num_nodes, num=sampling_steps + 1, dtype=np.float32).astype(np.int32)
    searching_edges_schedule = np.linspace(num_kept_edges, num_nodes, num=sampling_steps + 1, dtype=np.float32).astype(np.int32)
    
    def inference_fn(seed: int, coords: np.ndarray, dist_mat: np.ndarray):
        # features: jax.Array = encode(coords)
        features_manager = DynamicAugment(encode, coords, augment_level=augment_level)
        storage: dict[int, np.ndarray] = {}
        def _single_run(r: int):
            generator = np.random.default_rng(seed + r * 77)
            insertion = TSPPartialInsertion(batch_size, num_nodes, num_workers=num_workers)
            # generation
            edges = insertion.init_edges()
            start_nodes = generator.integers(0, num_nodes, size=[batch_size], dtype=np.int32)
            for i in range(sampling_steps):
                candidate_edges = decode_step(features_manager(generator), edges, start_nodes)
                insertion.insert(
                    edges, np.array(candidate_edges), 
                    generation_edges_schedule[i + 1] - generation_edges_schedule[i],
                )
            sols = insertion.get_sols()
            sols = tsp_two_opt(
                sols, dist_mat, two_opt_steps, 
                num_workers=num_workers, 
            )
            costs = get_costs_cpu(sols, dist_mat)
            # searching
            min_costs = costs
            for _ in range(cycles - 1):
                assert False
                edges = sols * num_nodes + np.roll(sols, shift=1, axis=-1)
                generator.permuted(edges, axis=1, out=edges)
                edges = np.stack(np.divmod(edges, num_nodes), axis=-1)
                edges[:, num_kept_edges:] = num_nodes * 2   # out-of-bound value
                insertion.set_state(np.ascontiguousarray(edges[:, :num_kept_edges]))
                for i in range(sampling_steps):
                    candidate_edges = decode_step(features_manager(generator), edges)
                    insertion.insert(
                        edges, np.array(candidate_edges), 
                        searching_edges_schedule[i + 1] - searching_edges_schedule[i],
                    )
                sols = insertion.get_sols()
                sols = tsp_two_opt(
                    sols, dist_mat, two_opt_steps, 
                    num_workers=num_workers, 
                )
                costs = get_costs_cpu(sols, dist_mat)
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
            seed + i * 888,
            coords[i * batch_size:(i + 1) * batch_size], 
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
        )
        return cost.mean().item()
    
    # triton autotune and warmup
    for _ in range(5):
        i = 0
        features = encode(coords[i * batch_size:(i + 1) * batch_size])
        candidate_edges = decode_step(
            features, 
            jnp.full([batch_size, num_nodes, 2], num_nodes * 2, dtype=jnp.int32),
            jnp.zeros([batch_size], dtype=jnp.int32),
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
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--two_opt_steps', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--heatmap_dtype', type=str, default='float32')
    parser.add_argument('--topk', type=eval, default=None)
    parser.add_argument('--augment_level', type=int, default=0)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: TSPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    dataset = dict(np.load(args.data))

    tsp_searching_decode(
        dataset, model, model_config,
        sampling_steps=args.sampling_steps, 
        cycles=args.cycles, keep_rate=args.keep_rate, 
        batch_size=args.batch_size,
        runs=args.runs,
        two_opt_steps=args.two_opt_steps,
        heatmap_dtype=args.heatmap_dtype,
        topk=args.topk,
        augment_level=args.augment_level,
        threads_over_batches=args.threads_over_batches,
        seed=args.seed,
    )
    
