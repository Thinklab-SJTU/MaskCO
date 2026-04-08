import numpy as np
import tensorflow as tf
from typing import Literal, Mapping, Callable
from lib import cvrp_random_two_opt


class CVRPDataloader:
    def __init__(
        self,
        datasets: Mapping[str, np.ndarray],
        batch_size: int,
        capacity: int | None = None,
        target_disruption: tuple[int, int] | None = None,
        need_current: bool = False,
        data_augment: Literal[0, 1, 2, 3] = 3,
        num_workers: int = 2,
    ) -> None:
        datasets = dict(datasets)
        assert {'coords', 'demands', 'routes'} <= set(datasets.keys()) <= {'coords', 'demands', 'routes', 'capacity'}
        assert capacity is not None or 'capacity' in datasets.keys()
        assert not (capacity is not None and 'capacity' in datasets.keys())
        self.coords: np.ndarray = datasets['coords']
        if capacity is not None:
            self.demands: np.ndarray = (datasets['demands'].astype(np.float64) / np.array(capacity, dtype=np.float64)).astype(np.float32)
        else:
            self.demands: np.ndarray = (datasets['demands'].astype(np.float64) / datasets['capacity'][..., None].astype(np.float64)).astype(np.float32)
        self.opt_routes: np.ndarray = datasets['routes']
        self.num_depots: int = np.equal(self.opt_routes[0], 0).astype(np.int32).sum().item()
        self.num_nodes: int = self.opt_routes.shape[1] - self.num_depots
        self.batch_size = batch_size
        self.steps_per_epoch = self.coords.shape[0] // batch_size
        self.num_workers = num_workers
        self.target_disruption = target_disruption
        self.need_current = need_current

        rot_choice = np.array([
            [[np.cos(i * np.pi / 2), -np.sin(i * np.pi / 2)], [np.sin(i * np.pi / 2), np.cos(i * np.pi / 2)]] for i in range(4)
        ], dtype=np.float32)
        def _argument_level_0(coords: np.ndarray, generator: np.random.Generator):
            return coords
        def _argument_level_1(coords: np.ndarray, generator: np.random.Generator):
            # random flip
            multiplier = (generator.integers(0, 2, size=(batch_size, 2), dtype=np.int32) * 2 - 1).astype(np.float32)
            coords = coords * multiplier[:, None, :]
            # random 90xdegree rot
            rot_mat = np.take(rot_choice, generator.integers(0, 4, size=(), dtype=np.int32), axis=0)
            coords = np.matmul(coords, rot_mat)
            return coords
        def _argument_level_2(coords: np.ndarray, generator: np.random.Generator):
            # random rot
            thetas = generator.uniform(0., 2 * np.pi, size=batch_size).astype(np.float32)
            sin_thetas = np.sin(thetas)
            cos_thetas = np.cos(thetas)
            rot_mat = np.stack([cos_thetas, -sin_thetas, sin_thetas, cos_thetas], axis=-1)
            rot_mat.resize((batch_size, 2, 2))
            return np.matmul(coords, rot_mat)
        def _argument_level_3(coords: np.ndarray, generator: np.random.Generator):
            return _argument_level_2(_argument_level_1(coords, generator), generator)
        
        self.argument_fn: Callable[[np.ndarray, np.random.RandomState], np.ndarray]
        self.argument_fn = eval(f'_argument_level_{data_augment}')

        int32_min = np.iinfo(np.int32).min
        int32_max = np.iinfo(np.int32).max
        current_base = np.array([0] * self.num_depots + [i for i in range(1, self.num_nodes + 1)], dtype=np.int32)
        current_base = np.stack([current_base] * batch_size, axis=0)
        assert current_base.shape[-1] == self.opt_routes.shape[-1]
        def preprocess(seed: int | np.ndarray, coords: np.ndarray, demands: np.ndarray, target: np.ndarray):
            # batched preprocess
            generator = np.random.default_rng(seed)
            
            coords = self.argument_fn(coords, generator)
            
            timestep = generator.uniform(0., 1., size=(batch_size,)).astype(np.float32)
            features = np.concat([coords, demands[..., None]], axis=-1)

            if need_current:
                current = generator.permutation(current_base, axis=-1)

            if target_disruption is not None:
                shuffle_times = generator.integers(*target_disruption, size=batch_size, dtype=np.int32)
                shuffle_key = generator.integers(int32_min, int32_max, size=batch_size, dtype=np.int32)
                noisy_target = np.stack([cvrp_random_two_opt(k, tar, t) for k, tar, t in zip(shuffle_key, target, shuffle_times)], axis=0)
                if need_current:
                    return features, target, noisy_target, current, timestep
                else:
                    return features, target, noisy_target, timestep
            else:
                if need_current:
                    return features, target, current, timestep
                else:
                    return features, target, timestep
        self.preprocess = preprocess
        
    @property
    def num_instances(self) -> int:
        return self.coords.shape[0]

    def __iter__(self):
        seeds = np.random.randint(0, np.iinfo(np.uint32).max, size=[self.num_instances], dtype=np.uint32)
        datasets = tf.data.Dataset.from_tensor_slices((seeds, self.coords, self.demands, self.opt_routes))
        datasets = datasets.shuffle(buffer_size=100, seed=np.random.randint(0, np.iinfo(np.uint32).max, size=tuple()))
        
        Tout = [tf.float32] + [tf.int32] * (3 - int(self.target_disruption is None) - int(not self.need_current)) + [tf.float32]
        datasets = datasets.batch(self.batch_size, drop_remainder=True)
        datasets = datasets.map(lambda x, y, z, w: tf.numpy_function(self.preprocess, inp=[x, y, z, w], Tout=Tout), num_parallel_calls=self.num_workers)
        return iter(datasets.as_numpy_iterator())
    
