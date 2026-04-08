import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
from models import MISModel, MISModelConfig
from training import load_ckpt
from helpers.mis_transform import edges2adj
from lib import MISPartialInsertion, mis_edges2neighbors, MISBatchedNeighbors


def mis_search(
    dataset: dict[str, np.ndarray],
    model: MISModel,
    sampling_steps: int,
    keep_rate: float,
    cycles: int,
    runs: int,
    batch_size: int,
    threads_over_batches: int | None,
    seed: int,
):
    num_workers = 1

    np.random.seed(seed)

    edges = dataset['edges']
    num_edges = np.any(edges != 0, axis=-1).astype(np.int32).sum(axis=-1)
    num_nodes = dataset['num_nodes']
    del dataset

    num_nodes_padded: int = num_nodes.max().item()
    num_instances = edges.shape[0]
    assert num_instances % (batch_size * threads_over_batches) == 0

    edges_list, num_edges_list, num_nodes_list = tuple(map(
        lambda x: np.split(x, num_instances // batch_size, axis=0),
        (edges, num_edges, num_nodes),
    ))
    del edges, num_edges, num_nodes

    adjmat_list = [
        edges2adj(edges, num_nodes_padded, dtype=jnp.float16)
        for edges in edges_list
    ]
    neighbors_list = [
        mis_edges2neighbors(
            edges, num_nodes, num_edges, tolist=False,
        ) for edges, num_nodes, num_edges in zip(edges_list, num_nodes_list, num_edges_list)
    ]

    @jax.jit
    def encode(
        adjmat: jax.Array,
        key: jax.Array,
    ):
        return model.encode(adjmat, key)

    @jax.jit
    def decode_step(
        features: jax.Array,
        current: jax.Array,
        adjmat: jax.Array,
        num_nodes: jax.Array,
        last_mis_size_pred: jax.Array,
    ):
        timestep = jnp.where(
            current.sum(axis=-1) == 0,
            0,
            current.sum(axis=-1) / last_mis_size_pred,
        )
        sigmoid_logits, softmax_logits = model.decode(features, current, timestep, adjmat, num_nodes)
        mis_size_pred = jax.nn.sigmoid(sigmoid_logits).sum(axis=-1)
        candidates_nodes = jnp.argsort(
            softmax_logits if softmax_logits is not None else sigmoid_logits, axis=-1, stable=False,
            descending=True,
        )
        return candidates_nodes, mis_size_pred
    
    def inference_fn(
        seed: int, num_nodes: np.ndarray,
        adjmat: jax.Array, neighbors: MISBatchedNeighbors,
        *, cycles: int,
    ):
        generator = np.random.default_rng(seed)
        features = encode(
            adjmat, 
            generator.integers(0, np.iinfo(np.uint32).max, size=[2], dtype=np.uint32)
        )
        num_nodes_np = num_nodes
        num_nodes = jax.device_put(num_nodes)
        storage: dict[int, np.ndarray] = {}
        def _single_run(r: int):
            insertion = MISPartialInsertion(
                neighbors, 
                num_nodes_np,
                num_nodes_padded,
                seed=seed + r * 77,
                num_workers=num_workers,
            )
            mis_size_pred = num_nodes   # init as `num_nodes`
            for i in range(sampling_steps):
                candidate_nodes, mis_size_pred = decode_step(
                    features, insertion.solution,
                    adjmat, num_nodes,
                    mis_size_pred,
                )
                candidate_nodes = np.array(candidate_nodes)             
                if i == sampling_steps - 1:
                    target_num_nodes = np.full(mis_size_pred.shape, -1, dtype=np.int32)    # insert as much as it can
                else:
                    target_num_nodes = (((i + 1) / sampling_steps) * np.array(mis_size_pred)).astype(np.int32)
                insertion.insert(candidate_nodes, target_num_nodes)

            sizes = insertion.num_inserted_nodes.copy()

            # searching
            best_sizes = sizes

            for _ in range(cycles - 1):
                insertion.partially_mask(keep_rate)
                for i in range(sampling_steps):
                    candidate_nodes, mis_size_pred = decode_step(
                        features, insertion.solution,
                        adjmat, num_nodes,
                        mis_size_pred,
                    )
                    candidate_nodes = np.array(candidate_nodes)             
                    if i == sampling_steps - 1:
                        target_num_nodes = np.full(mis_size_pred.shape, -1, dtype=np.int32)    # insert as much as it can
                    else:
                        target_num_nodes = (((i + 1) / sampling_steps * (1 - keep_rate) + keep_rate) * np.array(mis_size_pred)).astype(np.int32)
                    insertion.insert(candidate_nodes, target_num_nodes)
                best_sizes = np.maximum(best_sizes, insertion.num_inserted_nodes)
            
            storage[r] = best_sizes

        threads = [threading.Thread(target=_single_run, args=[r]) for r in range(runs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        best_sizes = storage[0]
        for i in range(1, runs):
            this_sizes = storage[i]
            best_sizes = np.maximum(best_sizes, this_sizes)
        return best_sizes
    
    def run_batch(i: int) -> float:
        sizes = inference_fn(
            i * 888,
            num_nodes_list[i],
            adjmat_list[i],
            neighbors_list[i],
            cycles=cycles,
        )
        return sizes.mean().item()
    
    # warmup and triton autotune
    inference_fn(
        1234,
        num_nodes_list[0],
        adjmat_list[0],
        neighbors_list[0],
        cycles=5, 
    )
    
    sizes: list[float] = []
    if threads_over_batches is None or threads_over_batches == 1:
        for i in tqdm.tqdm(range(num_instances // batch_size)):
            batch_cost = run_batch(i)
            print(f'batch {i} cost mean: {batch_cost:.6f}')
            sizes.append(batch_cost)
    else:
        assert num_instances % (batch_size * threads_over_batches) == 0
        with ThreadPoolExecutor(max_workers=threads_over_batches) as pool:
            for j in tqdm.tqdm(range(num_instances // batch_size // threads_over_batches)):
                batch_id_start = j * threads_over_batches
                batch_id_end = batch_id_start + threads_over_batches
                submit_sizes = list(pool.map(run_batch, range(batch_id_start, batch_id_end)))
                for i, batch_size in zip(range(batch_id_start, batch_id_end), submit_sizes):
                    print(f'batch {i} size mean: {batch_size:.6f}')
                    sizes.append(batch_size)

    mean_size = sum(sizes) / len(sizes)
    print(f'Avg Size: {mean_size:.6f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sampling_steps', type=int, required=True)
    parser.add_argument('--cycles', type=int, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--keep_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--threads_over_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    params, _, _, model_config, _, _ = load_ckpt(args.ckpt)
    model_config: MISModelConfig
    model = model_config.construct_model()
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)
    dataset = dict(np.load(args.data))

    assert model_config.sigmoid_head or args.sampling_steps == 1

    mis_search(
        dataset, model,
        args.sampling_steps,
        args.keep_rate,
        args.cycles,
        args.runs,
        args.batch_size,
        args.threads_over_batches,
        args.seed,
    )
