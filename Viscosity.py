import warnings
warnings.simplefilter('ignore')
import jax.numpy as jnp
from jax import jit, vmap

from jaxMPS.Weight import *

@jit
def acc_viscosity(vi, vj, w):
    return w*(vj - vi)

vmap_acc_viscosity = vmap(acc_viscosity, in_axes = (None, 0, 0))
vmap_acc_viscosity = jit(vmap(vmap_acc_viscosity, in_axes = (0, None, 0)))

@jit
def getAcc_viscosity(coef, Xi, Xj, Vi, Vj, re):
    W = vmap_weight(Xi, Xj, re)
    return coef * jnp.sum(vmap_acc_viscosity(Vi, Vj, W), axis = 1)