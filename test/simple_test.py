import warnings
warnings.simplefilter('ignore')
import numpy as np
import jaxMPS as mps

particle_distance = 0.025 #[m]
dt = 0.001 #時間刻み幅 [s]

fluid_position, fluid_velocity, wall_position, wall_velocity, dummy_position, dummy_velocity \
= mps.data.init_dambreak(particle_distance)

field = mps.field2d(particle_distance, fluid_position, wall_position, dummy_position)
field.step(dt)

field.save(field.wall_position, "./vtk/wall.vtk")
for itr in range(1000):
    print(itr)
    if (itr % 10) == 0:
        field.save(field.fluid_position, "./vtk/fluid"+str(itr).zfill(5)+".vtk")
    field.step(dt)