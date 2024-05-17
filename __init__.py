import warnings
warnings.simplefilter('ignore')
import jax
import jax.numpy as jnp
from jax import jit, vmap
import pyvista as pv
import numpy as np
import sys

from jaxMPS import data
from jaxMPS.Weight import *
from jaxMPS.Viscosity import *
from jaxMPS.Collision import *
from jaxMPS.Pressure import *




class field2d:
    def __init__(self,
                particle_distance,
                fluid_position,
                wall_position,
                dummy_position,
                fluid_velocity = None,
                wall_velocity = None,
                dummy_velocity = None,
                rho0 = 1000.,
                nu = 1e-6):
        
        self.dim = 2
        self.rho0 = rho0
        self.nu = nu
        self.particle_distance = particle_distance

        self.fluid_position = fluid_position
        self.wall_position = wall_position
        self.dummy_position = dummy_position

        self.fluid_velocity = jnp.zeros(fluid_position.shape) if fluid_velocity is None else fluid_velocity
        self.wall_velocity = jnp.zeros(wall_position.shape) if wall_velocity is None else wall_velocity
        self.dummy_velocity = jnp.zeros(dummy_position.shape) if dummy_velocity is None else dummy_velocity

        g1 = jnp.zeros(len(fluid_position)); g2 = -9.8*jnp.ones(len(fluid_position))
        self.gravity = jnp.stack((g1, g2), axis = -1)

        self.r4_numDensity = 2.1*particle_distance
        self.r4_grad = 2.1*particle_distance
        self.r4_laplacian = 3.1*particle_distance
        self.n04_numDensity, self.n04_grad, self.n04_laplacian, self.lam = self.getN0_lambda()

        self.threshold_ratio_num_density = 0.97
        self.coef_restitution = 0.2
        self.pressure_relaxation_coef = 0.2
        self.compressibility = 4.5e-10

        self.collision_distance = 0.5*particle_distance
    


    def getN0_lambda(self):
        n04_numDensity = 0.; n04_grad = 0.; n04_laplacian = 0.; lam = 0.

        zero = jnp.zeros(2)
        for ix in range(-4, 5):
            for iy in range(-4,5):
                xi = self.particle_distance*jnp.array([ix, iy])
                r2 = jnp.linalg.norm(xi)**2

                n04_numDensity += weight(xi, zero, self.r4_numDensity)
                n04_grad += weight(xi, zero, self.r4_grad)
                n04_laplacian += weight(xi, zero, self.r4_laplacian)
                lam += r2*weight(xi, zero, self.r4_laplacian)
        
        lam /= n04_laplacian
        return n04_numDensity, n04_grad, n04_laplacian, lam
    
    def Acc_viscosity(self):
        coef = self.nu*(2.*self.dim) / self.n04_laplacian / self.lam

        ###流体間の粘性応力計算
        Acc_fluid_fluid = \
        getAcc_viscosity(coef, self.fluid_position, self.fluid_position, self.fluid_velocity, self.fluid_velocity, self.r4_laplacian)
        ###流体壁面間の粘性応力計算
        Acc_fluid_wall = \
        getAcc_viscosity(coef, self.fluid_position, self.wall_position, self.fluid_velocity, self.wall_velocity, self.r4_laplacian)
        ###流体ダミー間の粘性応力計算
        Acc_fluid_dummy = \
        getAcc_viscosity(coef, self.fluid_position, self.dummy_position, self.fluid_velocity, self.dummy_velocity, self.r4_laplacian)

        return Acc_fluid_fluid + Acc_fluid_wall + Acc_fluid_dummy
    
    def fix_Collision(self):
        vdiff_fluid = \
        fixCollision(self.fluid_position, self.fluid_position, self.fluid_velocity, self.fluid_velocity, \
        self.collision_distance, self.coef_restitution)

        vdiff_wall = \
        fixCollision(self.fluid_position, self.wall_position, self.fluid_velocity, self.wall_velocity, \
        self.collision_distance, self.coef_restitution)

        vdiff_dummy = \
        fixCollision(self.fluid_position, self.dummy_position, self.fluid_velocity, self.dummy_velocity, \
        self.collision_distance, self.coef_restitution)

        return vdiff_fluid + vdiff_wall + vdiff_dummy


    def step(self, dt):
        acc = self.gravity + self.Acc_viscosity()

        self.fluid_velocity += acc*dt
        self.fluid_position += self.fluid_velocity*dt

        velocity_diff = self.fix_Collision()
        self.fluid_position += velocity_diff*dt
        self.fluid_velocity += velocity_diff

        pressure = getPressure(dt, self.dim, self.rho0,
                               self.fluid_position, self.wall_position, self.dummy_position, 
                               self.r4_numDensity, self.r4_laplacian, self.n04_numDensity, self.n04_laplacian, self.lam, 
                               self.threshold_ratio_num_density, self.pressure_relaxation_coef, self.compressibility)
        
        min_pressure = getMinPressure(pressure, self.fluid_position, self.wall_position, self.r4_grad)

        acc = getAcc_pressure(self.fluid_position, self.wall_position, pressure, min_pressure, \
                                self.dim, self.n04_grad, self.r4_grad, self.rho0)
        
        self.fluid_velocity += acc*dt
        self.fluid_position += acc*(dt**2)
    

    def save(self, position, filename):
        zero = np.zeros((len(position), 1))
        position = np.concatenate((jnp.asarray(position), zero), axis = 1)
        points = pv.PolyData(position)
        points.save(filename)