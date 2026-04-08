import jax
import jax.numpy as jnp


def sol2adj(
    sols: jax.Array, dtype: jax.typing.DTypeLike = jnp.float32, 
    num_nodes: int | None = None,
    is_cvrp: bool = False,
):  
    '''
    :param num_nodes: Number of total nodes, including the depot.
    '''

    if is_cvrp:
        assert num_nodes is not None
    if num_nodes is None:
        num_nodes = sols.shape[-1]
    def _unbatched_impl(sol: jax.Array):
        out = jnp.zeros([num_nodes, num_nodes], dtype=dtype)
        sol_shifted = jnp.roll(sol, shift=1)
        out = out.at[sol, sol_shifted].set(1, mode='drop')
        if is_cvrp:
            out = out.at[sol_shifted, sol].add(1, mode='drop')
            out = out.at[0, 0].set(0)
        else:
            out = out.at[sol_shifted, sol].set(1, mode='drop')
        return out
    return jax.vmap(_unbatched_impl, in_axes=0, out_axes=0)(sols)


def sol2adj_with_mask(
    sols: jax.Array, mask: jax.Array, 
    dtype: jax.typing.DTypeLike = jnp.float32, 
    num_nodes: int | None = None,
    is_cvrp: bool = False,
):
    '''
    :param num_nodes: Number of total nodes, including the depot.
    '''

    if is_cvrp:
        assert num_nodes is not None
    if num_nodes is None:
        num_nodes = sols.shape[-1]
    def _unbatched_impl(sol: jax.Array, mask: jax.Array):
        assert sol.ndim == mask.ndim == 1
        assert mask.shape[-1] == sol.shape[-1]
        values = mask.astype(dtype)

        out = jnp.zeros([num_nodes, num_nodes], dtype=dtype)
        sol_shifted = jnp.roll(sol, shift=1)
        out = out.at[sol, sol_shifted].set(values, mode='drop')
        if is_cvrp:
            out = out.at[sol_shifted, sol].add(values, mode='drop')
            out = out.at[0, 0].set(0)
        else:
            out = out.at[sol_shifted, sol].set(values, mode='drop')
        return out
    return jax.vmap(_unbatched_impl, in_axes=(0, 0), out_axes=0)(sols, mask)
    

def edges2adj(
    edges: jax.Array, 
    dtype: jax.typing.DTypeLike = jnp.float32,
    num_nodes: int | None = None,
    is_cvrp: bool = False,
):
    '''
    :param num_nodes: Number of total nodes, including the depot.
    '''
    
    if is_cvrp:
        assert num_nodes is not None
        raise NotImplementedError()
    if num_nodes is None:
        num_nodes = edges.shape[-2]
    def _unbatched_impl(edges: jax.Array):
        adjmat = jnp.zeros([num_nodes, num_nodes], dtype=dtype)
        adjmat = adjmat.at[edges[:, 0], edges[:, 1]].set(1, mode='drop')
        adjmat = adjmat.at[edges[:, 1], edges[:, 0]].set(1, mode='drop')
        return adjmat
    return jax.vmap(_unbatched_impl, in_axes=0, out_axes=0)(edges)
