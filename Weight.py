import warnings
warnings.simplefilter('ignore')
import jax.numpy as jnp
import jax
from jax import jit, vmap

@jit
def weight(xi, xj, re):
    r = jnp.linalg.norm(xi - xj)
    return jnp.where(r == 0., 0., jnp.clip((re/r) - 1.0, a_min = 0.))

vmap_weight = vmap(weight, in_axes = (None, 0, None))
vmap_weight = jit(vmap(vmap_weight, in_axes = (0, None, None)))