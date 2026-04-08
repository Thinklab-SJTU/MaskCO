import jax
import jax.numpy as jnp

from typing import Literal 


def rotate(inputs: jax.Array, theta: float | jax.Array) -> jax.Array:
    rot_mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]], dtype='float32')
    return jnp.dot(inputs.astype('float32'), rot_mat, precision=jax.lax.Precision.HIGHEST).astype(inputs.dtype)


def get_rot_mat(theta: float | jax.Array, dtype: jnp.dtype = 'float32') -> jax.Array:
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]], dtype=dtype)


def reflect_x(inputs: jax.Array) -> jax.Array:
    return inputs.at[..., 0].set(- inputs[..., 0])


def reflect_y(inputs: jax.Array) -> jax.Array:
    return inputs.at[..., 1].set(- inputs[..., 1])


def median(xs: jax.Array) -> jax.Array:
    xs = xs.sort(axis=-1)
    d = xs.shape[-1]
    return xs[..., d // 2]


def normalize(inputs: jax.Array, centering_method: Literal['mean', 'median'] = 'mean') -> jax.Array:
    if centering_method == 'mean':
        inputs = inputs - inputs.mean(axis=-2, keepdims=True)
    elif centering_method == 'median':
        inputs = inputs - jax.vmap(median, in_axes=-1, out_axes=-1)(inputs)[:, None]
    else:
        raise

    # scale by the mean L2-norm of all nodes
    inputs_norm = jnp.linalg.norm(inputs, axis=-1, keepdims=True, ord=2)
    inputs_norm_mean = inputs_norm.mean(axis=-2, keepdims=True)
    inputs = inputs / inputs_norm_mean

    return inputs


def get_normalize_scale_rate(inputs: jax.Array) -> jax.Array:
    inputs = inputs - inputs.mean(axis=-2, keepdims=True)

    # scale by the mean L2-norm of all nodes
    inputs_norm = jnp.linalg.norm(inputs, axis=-1, keepdims=True, ord=2)
    inputs_norm_mean = inputs_norm.mean(axis=-2, keepdims=False).reshape(-1,)

    return 1. / inputs_norm_mean


def cdist(coords: jax.Array) -> jax.Array:
    if coords.ndim == 2:
        return jax.jit(_cdist_unbatched)(coords)
    elif coords.ndim == 3:
        return jax.vmap(_cdist_unbatched)(coords)


def cdist_mm(coords: jax.Array) -> jax.Array:
    if coords.ndim == 2:
        return jax.jit(_cdist_unbatched_mm_impl)(coords)
    elif coords.ndim == 3:
        return jax.vmap(_cdist_unbatched_mm_impl)(coords)
    

def _cdist_unbatched(coord: jax.Array) -> jax.Array:
    coord_diff = coord[:, None, :] - coord[None, :, :]
    return jnp.linalg.norm(coord_diff, axis=-1, ord=2)


def _cdist_unbatched_mm_impl(coord: jax.Array) -> jax.Array:
    x_square = jnp.sum(coord * coord, axis=1)
    xy = jnp.dot(coord, coord.T)
    return jnp.sqrt(jnp.maximum(x_square[:, None] + x_square - 2 * xy, 0)) 

