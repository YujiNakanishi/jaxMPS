import warnings
warnings.simplefilter('ignore')
import jax.numpy as jnp

def init_dambreak(particle_distance, eps = None, width = 1., height = 0.6):
    if eps is None: eps = 0.01*particle_distance

    nx = int(width/particle_distance); ny = int(height/particle_distance)

    fluid_position = []
    wall_position = []
    dummy_position = []

    for ix in range(-4, nx + 5):
        for iy in range(-4, ny + 5):
            x = particle_distance*ix
            y = particle_distance*iy

            p_type = "empty"

            if (x > -4.*particle_distance + eps) and \
               (x <= width + 4.*particle_distance + eps) and \
               (y > -4.*particle_distance + eps) and \
               (y <= height + eps):
               p_type = "dummy"
            
            if (x > -2.*particle_distance + eps) and \
               (x <= width + 2.*particle_distance + eps) and \
               (y > -2.*particle_distance + eps) and \
               (y <= height + eps):
               p_type = "wall"
            
            if (x > -4.*particle_distance + eps) and \
               (x <= width + 4.*particle_distance + eps) and \
               (y > height - 2.*particle_distance + eps) and \
               (y < height + eps):
               p_type = "wall"

            if (x > eps) and (x <= width + eps) and (y > eps):
                p_type = "empty"
            
            if (x > eps) and (x <= width/4. + eps) and (y > eps) and (y <= (5./6.)*height + eps):
                p_type = "fluid"
            

            if p_type == "fluid":
                fluid_position.append([x, y])
            elif p_type == "wall":
                wall_position.append([x, y])
            elif p_type == "dummy":
                dummy_position.append([x, y])
    
    fluid_position = jnp.array(fluid_position)
    fluid_velocity = jnp.zeros(fluid_position.shape)
    wall_position = jnp.array(wall_position)
    wall_velocity = jnp.zeros(wall_position.shape)
    dummy_position = jnp.array(dummy_position)
    dummy_velocity = jnp.zeros(dummy_position.shape)

    return fluid_position, fluid_velocity, wall_position, wall_velocity, dummy_position, dummy_velocity