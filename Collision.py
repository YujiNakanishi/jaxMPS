import warnings
warnings.simplefilter('ignore')
import jax.numpy as jnp
from jax import jit, vmap

@jit
def collision(xi, xj, vi, vj, collision_distance, coef_restitution):
    r = jnp.linalg.norm(xi - xj)
    f = 0.5*(1. + coef_restitution)
    acc = jnp.where(r < collision_distance, jnp.sum((vi -vj)*(xj - xi)/r), 0.)

    return jnp.where(acc > 0., -f*(xj - xi)/r, 0.)


vmap_collision = vmap(collision, in_axes = (None, 0, None, 0, None, None))
vmap_collision = jit(vmap(vmap_collision, in_axes = (0, None, 0, None, None, None)))

@jit
def fixCollision(Xi, Xj, Vi, Vj, collision_distance, coef_restitution):
    return jnp.sum(vmap_collision(Xi, Xj, Vi, Vj, collision_distance, coef_restitution), axis = 1)