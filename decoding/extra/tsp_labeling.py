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
from helpers import edges2adj
from training import load_ckpt
from lib import tsp_two_opt, TSPPartialInsertion, tsp_eval_cost
from decoding.utils import (
    convert_heatmap_dtype, 
    CoordDynamicAugment as DynamicAugment,
)
from decoding.gpu_heuristics.tsp import two_opt as gpu_tsp_two_opt
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
    np.random.seed(seed)

    num_workers = 1

    assert {'coords'} <= set(dataset.keys()) <= {'coords', 'opt_tours', 'opt_costs'}

    coords = dataset['coords']
    dist_mat_np = np.array(jax.jit(cdist)(coords))

    num_instances, num_nodes, _ = coords.shape
    num_nodes: int

    jitted_gpu_two_opt = jax.jit(partial(gpu_tsp_two_opt, num_steps=two_opt_steps))

    def _encode(raw_features: jax.Array):
        features = normalize(raw_features, centering_method='mean')
        features = model.encode(features)
        return features

    def _decode_step(
        features: jax.Array, 
        edges: jax.Array,
    ):
        timestep = (edges < num_nodes).sum(axis=[-1, -2]).astype(jnp.float32) / (2 * num_nodes)
        adjmat = edges2adj(edges, dtype=jnp.int8)
        logits = model.decode(features, timestep, adjmat.astype(jnp.float16))
        adjmat_pred = jax.nn.softmax(logits, axis=-1) * 2
        adjmat_pred = jnp.where(adjmat, 0., adjmat_pred)
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
        dist_mat_gpu = jnp.array(dist_mat)
        def _single_run(r: int):
            generator = np.random.default_rng(seed + r * 77)
            insertion = TSPPartialInsertion(batch_size, num_nodes, num_workers=num_workers)
            # generation
            edges = insertion.init_edges()
            for i in range(sampling_steps):
                candidate_edges = decode_step(features_manager(generator), edges)
                insertion.insert(
                    edges, np.array(candidate_edges), 
                    generation_edges_schedule[i + 1] - generation_edges_schedule[i],
                )
            sols = insertion.get_sols()
            sols = jitted_gpu_two_opt(
                sols, dist_mat_gpu,
            )
            sols = np.array(sols)
            costs = get_costs_cpu(sols, dist_mat)
            # searching
            min_costs = costs
            sols_bsf = sols
            for _ in range(cycles - 1):
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
                sols = jitted_gpu_two_opt(
                    sols, dist_mat_gpu,
                )
                sols = np.array(sols)
                costs = get_costs_cpu(sols, dist_mat)
                sols_bsf = np.where(
                    (costs < min_costs)[:, None],
                    sols, sols_bsf,
                )
                min_costs = np.minimum(min_costs, costs)
            storage[r] = (min_costs, sols_bsf)
      
        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        min_costs, best_tours = storage[0]
        for i in range(1, runs):
            this_costs, this_tours = storage[i]
            best_tours = np.where(
                (this_costs < min_costs)[:, None],
                this_tours,
                best_tours,                    
            )
            min_costs = np.minimum(min_costs, this_costs)
        return min_costs, best_tours
    
    def run_batch(i: int) -> float:
        cost, tours = inference_fn(
            seed + i * 888,
            coords[i * batch_size:(i + 1) * batch_size], 
            dist_mat_np[i * batch_size:(i + 1) * batch_size],
        )
        return cost.mean().item(), tours
    
    # triton autotune and warmup
    for _ in range(5):
        i = 0
        features = encode(coords[i * batch_size:(i + 1) * batch_size])
        candidate_edges = decode_step(
            features, 
            jnp.full([batch_size, num_nodes, 2], num_nodes * 2, dtype=jnp.int32),
        )
        del features, candidate_edges
    
    costs: list[float] = []
    tours: list[np.ndarray] = []
    assert num_instances % (batch_size * threads_over_batches) == 0
    with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
        for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
            batch_id_start = j * threads_over_batches
            batch_id_end = batch_id_start + threads_over_batches
            submit_results = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
            for i, (batch_cost, batch_tours) in zip(range(batch_id_start, batch_id_end), submit_results):
                print(f'batch {i} cost mean: {batch_cost:.6f}')
                costs.append(batch_cost)
                tours.append(batch_tours)
    mean_cost = sum(costs) / len(costs)
    print(f'mean cost: {mean_cost:.6f}')
    
    tours = np.concatenate(tours, axis=0)

    return mean_cost, tours



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True)
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
    # parser.add_argument('--seed', type=int, default=0)


    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_nodes', type=int, required=True)
    parser.add_argument('--num_instances', type=int, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--submit_batch_size', type=int, required=True)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: TSPModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    # dataset = dict(np.load(args.data))

    generator = np.random.default_rng(args.seed)
    num_instances: int = args.num_instances
    num_nodes: int = args.num_nodes
    save_path: str = args.save_path
    submit_batch_size: int = args.submit_batch_size
    import os
    assert not os.path.exists(save_path) 
    assert num_instances % submit_batch_size == 0

    coords = generator.uniform(0., 1., size=[num_instances, num_nodes, 2]).astype(np.float32)
    tours = []
    costs = []

    for i in range(num_instances // submit_batch_size):
        submit_cost, submit_tours = tsp_searching_decode(
            dict(coords=coords[i * submit_batch_size:(i + 1) * submit_batch_size]), 
            model, model_config,
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
        tours.append(submit_tours)
        costs.append(submit_cost)

    tours = np.concatenate(tours, axis=0)
    mean_cost = sum(costs) / len(costs)
    print(f'Labeled Mean Cost: {mean_cost:.6f}')

    np.savez_compressed(
        save_path,
        coords=coords, opt_tours=tours,
    )        
    
