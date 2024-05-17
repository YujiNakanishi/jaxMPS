import warnings
warnings.simplefilter('ignore')
import jax.numpy as jnp
from jax import jit, vmap
import copy
import sys

from jaxMPS.Weight import *


def getPressure(dt, dim, rho0,
                fluid_position, wall_position, dummy_position, 
                r04numDensity, r04laplacian, n04numDensity, n04laplacian, lam,
                threshold_ratio_num_density, pressure_relaxation_coef, compressibility):
    #####数密度の計算
    W_fluid_fluid = vmap_weight(fluid_position, fluid_position, r04numDensity)
    W_fluid_wall = vmap_weight(fluid_position, wall_position, r04numDensity)
    W_fluid_dummy = vmap_weight(fluid_position, dummy_position, r04numDensity)
    W_wall_wall = vmap_weight(wall_position, wall_position, r04numDensity)
    W_wall_dummy = vmap_weight(wall_position, dummy_position, r04numDensity)

    numDensity_fluid = jnp.sum(W_fluid_fluid, axis = 1) + jnp.sum(W_fluid_wall, axis = 1) + jnp.sum(W_fluid_dummy, axis = 1)
    numDensity_wall = jnp.sum(W_fluid_wall, axis = 0) + jnp.sum(W_wall_wall, axis = 1) + jnp.sum(W_wall_dummy, axis = 1)
    numDensity = jnp.concatenate((numDensity_fluid, numDensity_wall))

    #####ソース項の計算
    is_surface = (numDensity < threshold_ratio_num_density * n04numDensity)
    source = jnp.where(is_surface, 0., pressure_relaxation_coef * ((numDensity - n04numDensity)/n04numDensity) / (dt**2))

    #####係数行列の計算
    coef = 2.*dim/rho0/lam/n04laplacian
    W_fluid_fluid = vmap_weight(fluid_position, fluid_position, r04laplacian)
    W_fluid_wall = vmap_weight(fluid_position, wall_position, r04laplacian)
    W_wall_wall = vmap_weight(wall_position, wall_position, r04laplacian)

    W1 = jnp.concatenate((-W_fluid_fluid, -W_fluid_wall), axis = 1)
    W2 = jnp.concatenate((-W_fluid_wall.T, -W_wall_wall), axis = 1)
    A = jnp.concatenate((W1, W2))*coef

    for j in range(len(A)):
        A = A.at[j,j].set(jnp.sum(-A[:,j]) + A[j,j] + compressibility/(dt**2))
    
    for idx, i_s in enumerate(is_surface):
        if i_s:
            a = jnp.zeros(len(A)); a = a.at[idx].set(1.)
            A = A.at[idx].set(a)

    pressure = jnp.linalg.solve(A, source)
    pressure = jnp.where(pressure > 0., pressure, 0.)

    return pressure


def dist(xi, xj):
    return jnp.linalg.norm(xi - xj)

vmap_dist = vmap(dist, in_axes = (None, 0))
vmap_dist = jit(vmap(vmap_dist, in_axes = (0, None)))
def getMinPressure(pressure, fluid_position, wall_position, re):
    X = jnp.concatenate((fluid_position, wall_position))
    neighbors = (vmap_dist(X, X) < re)[:len(fluid_position)]

    min_pressure = jnp.array([jnp.min(pressure[n]) for n in neighbors])

    return min_pressure


def acc_pressure(xi, xj, min_pi, pj, re):
    w = weight(xi, xj, re)
    r = jnp.linalg.norm(xi - xj)
    acc = jnp.where(r == 0., 0., (xj - xi)*(pj - min_pi)*w/(r**2))

    return acc

vmap_acc_pressure = vmap(acc_pressure, in_axes = (None, 0, None, 0, None))
vmap_acc_pressure = jit(vmap(vmap_acc_pressure, in_axes = (0, None, 0, None, None)))

@jit
def getAcc_pressure(fluid_position, wall_position, pressure, min_pressure, dim, n0, re, rho0):
    fluid_pressure = pressure[:len(fluid_position)]
    wall_pressure = pressure[len(fluid_position):]

    coef = dim/n0/rho0
    Acc_fluid = jnp.sum(vmap_acc_pressure(fluid_position, fluid_position, min_pressure, fluid_pressure, re), axis = 1)
    Acc_wall = jnp.sum(vmap_acc_pressure(fluid_position, wall_position, min_pressure, wall_pressure, re), axis = 1)

    Acc = -(Acc_fluid + Acc_wall)*coef

    return Acc