import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# params
q = 1.0      # charge of the particle
B = 1.0      # magnetic field strength
m = 1.0      # mass of the particle
b = 0.1      # drag coefficient

omega = q * B / m  # cyclotron frequency
lambda_ = b / m     # damping coefficient

# eqs of motion
def equations(t, y):
    vx, vy, vz, x, y, z = y
    dvx_dt = omega * vy - lambda_ * vx
    dvy_dt = -omega * vx - lambda_ * vy
    dvz_dt = -lambda_ * vz
    dx_dt = vx
    dy_dt = vy
    dz_dt = vz
    return [dvx_dt, dvy_dt, dvz_dt, dx_dt, dy_dt, dz_dt]

# initial conditions
vx0, vy0, vz0 = 1.0, 0.0, 1.0  # initial velocities
x0, y0, z0 = 0.0, 0.0, 0.0      # initial positions

# t span
t_span = (0, 50)
t_eval = np.linspace(0, 50, 5000)  # time points for evaluation
initial_conditions = [vx0, vy0, vz0, x0, y0, z0]
solution = solve_ivp(equations, t_span, initial_conditions, t_eval=t_eval)

# extract results
t = solution.t
x, y, z = solution.y[3], solution.y[4], solution.y[5]

# animate
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Motion of a Charged Particle in a Uniform Magnetic Field")

# axis limits
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
ax.set_zlim([min(z), max(z)])

line, = ax.plot([], [], [], 'r', label='Particle Trajectory')
dot, = ax.plot([], [], [], 'ro')
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
ax.legend()

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    dot.set_data([], [])
    dot.set_3d_properties([])
    time_text.set_text('')
    return line, dot, time_text

def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(z[:frame])
    dot.set_data(x[frame], y[frame])
    dot.set_3d_properties(z[frame])
    time_text.set_text(f'Time: {t[frame]:.2f} s')
    return line, dot, time_text

ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False, interval=2)
plt.show()
