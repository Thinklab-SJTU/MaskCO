import jax
import jax.numpy as jnp
from flax import nnx
from flax.jax_utils import replicate
from models import CVRPModelConfig, CVRPModel
from modules.functional import coord_normalize
from training.TrainConfig import TrainConfig
from training import Logger, load_ckpt, save_ckpt
from data import CVRPDataloader
from helpers import sol2adj, sol2adj_with_mask
import optax
import argparse
import numpy as np
from functools import partial


def train_cvrp(
    dataloader: CVRPDataloader, model_config: CVRPModelConfig, train_config: TrainConfig,
    model: CVRPModel, opt_state: optax.OptState | None, 
    save_interval: int, logdir: str, savedir: str,
    step: int = 0,
):
    tx = train_config.init_opimizer()
    graphdef, params = nnx.split(model)
    if opt_state is None:
        opt_state = tx.init(params)
    num_nodes = train_config.num_nodes
    assert train_config.target_disruption is None
    logger = Logger(logdir, step=step)

    num_devices = jax.device_count()
    assert train_config.batch_size % num_devices == 0
    is_replicated = num_devices > 1
    batch_size_per_device = train_config.batch_size // num_devices
    if is_replicated:
        params = replicate(params)
        opt_state = replicate(opt_state)
    
    @partial(jax.jit, donate_argnums=[1, 2])
    def train_step(
        graphdef: nnx.GraphDef[CVRPModel], params: nnx.State, opt_state: optax.OptState,
        raw_features: jax.Array, target: jax.Array, timestep: jax.Array,
        key: jax.Array,
    ):
        raw_features = raw_features.at[..., :2].set(coord_normalize(raw_features[..., :2]))

        route_len = target.shape[-1]
        tgt_adjmat = sol2adj(target, dtype=jnp.float32, is_cvrp=True, num_nodes=num_nodes + 1)

        current = target
        keep_prob = timestep
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(
            subkey, 
            keep_prob.reshape(batch_size_per_device, 1), 
            shape=(batch_size_per_device, route_len),
        )
        cur_adjmat = sol2adj_with_mask(current, mask=mask, dtype=jnp.float16, is_cvrp=True, num_nodes=num_nodes + 1)

        def loss_fn(params: nnx.State):
            model = nnx.merge(graphdef, params)
            features = model.encode(raw_features)
            logits = model.decode(features, timestep, cur_adjmat)
            logits = jax.nn.log_softmax(logits)
            return - (logits[:, 1:] * tgt_adjmat[:, 1:]).mean() * ((num_nodes + 1) / 2)     # Should the model be required to predict the known part? 
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params)
        if is_replicated:
            grads = jax.lax.pmean(grads, axis_name='data')
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state, key
    
    if not is_replicated:
        train_step = jax.jit(train_step, donate_argnums=[1, 2])
    else:
        train_step = jax.pmap(train_step, donate_argnums=[1, 2], axis_name='data')
    
    while True:
        key = jax.random.wrap_key_data(np.random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, size=[2], dtype=np.uint32))
        if num_devices > 1:
            key = jax.random.split(key, num_devices)
        for raw_features, target, timestep in dataloader:
            step += 1

            if is_replicated:
                raw_features, target, timestep = tuple(map(
                    lambda x: x.reshape((num_devices, batch_size_per_device) + x.shape[1:]),
                    (raw_features, target, timestep),
                ))
            
            loss, params, opt_state, key = train_step(graphdef, params, opt_state, raw_features, target, timestep, key)
            logger(**{'loss/loss': loss})

            if step % save_interval == 0:
                save_ckpt(
                    params, opt_state, np.random.get_state(),
                    model_config, train_config, step, savedir,
                    is_replicated=is_replicated,
                )



if __name__ == '__main__':
    np.random.seed(42)

    from helpers import with_invalid_kwargs_filtered, maybe_eval
    parser = argparse.ArgumentParser()
    # train config
    parser.add_argument('--num_nodes', type=int, required=True)     # e.g. CVRP100 -> 100
    parser.add_argument('--capacity', type=eval, required=True)
    parser.add_argument('--num_steps', type=int, default=10 ** 6)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--peak_lr', type=float, default=1e-3)
    parser.add_argument('--end_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_type', type=str, default='randperm')
    parser.add_argument('--target_disruption', type=partial(maybe_eval, should_keep=['default']), default=None)

    parser.add_argument('--optimizer_type', type=str, default='adamw')

    parser.add_argument('--save_interval', type=int, default=10 ** 10)  # default value leads to never save
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--savedir', type=str, default=None)
    
    parser.add_argument('--model_config', type=str, default='')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--data', type=str, required=True)
    
    parser.add_argument('--data_augment', type=int, default=3)

    parser.add_argument('--ignore_ckpt_train_config', action='store_true', default=False)

    args = parser.parse_args()

    
    
    params, opt_state, np_rd_state, model_config, train_config, step = load_ckpt(args.ckpt)
    if args.ignore_ckpt_train_config:
        opt_state, np_rd_state, train_config, step = None, None, None, 0
    if np_rd_state is not None:
        np.random.set_state(np_rd_state)
    if train_config is None:
        train_config = with_invalid_kwargs_filtered(TrainConfig)(**vars(args))
    if model_config is None:
        model_config = CVRPModelConfig.get_config(args.model_config)
    model = model_config.construct_model()
    if params is not None:
        graphdef = nnx.graphdef(model)
        model = nnx.merge(graphdef, params)
    dataloader = CVRPDataloader(
        np.load(args.data),
        batch_size=train_config.batch_size,
        capacity=args.capacity,
        target_disruption=None,
        need_current=False,
        data_augment=args.data_augment,
        num_workers=2,
    )
    
    train_cvrp(
        dataloader, model_config, train_config, model, opt_state, 
        save_interval=args.save_interval,
        logdir=args.logdir, savedir=args.savedir,
        step=step,
    )

