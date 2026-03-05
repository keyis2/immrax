#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import immrax as irx

device = "gpu"
method = "euler"

class Vehicle(irx.OpenLoopSystem):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 4

    def f(
        self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        px, py, psi, v = x.ravel()
        u1, u2 = u.ravel()
        beta = jnp.arctan(jnp.tan(u2) / 2)
        return jnp.array(
            [v * jnp.cos(psi + beta), v * jnp.sin(psi + beta), v * jnp.sin(beta), u1]
        )


olsys = Vehicle()

net = irx.NeuralNetwork("100r100r2")
clsys = irx.NNCSystem(olsys, net)

clembsys = irx.NNCEmbeddingSystem(clsys, "crown", "local", "local")
permutations = irx.standard_permutation(1 + 4 + 2 + 1)
corners = irx.two_corners(1 + 4 + 2 + 1)

# Returns [7.95,8.05] x [6.95,7.05] x [-2pi/3 - 0.01, -2pi/3 + 0.01] x [1.99, 2.01]
x0 = irx.icentpert([8, 7, -2 * jnp.pi / 3, 2], [0.05, 0.05, 0.01, 0.01])
w = irx.icentpert([0.0], 0.0)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from immrax.utils import get_partitions_ut, run_times, draw_iarrays, gen_ics
from immutabledict import immutabledict


def compute_and_plot(ax, N, solver, device):
    # Computing the Reachable Set of the Closed-Loop Embedding System
    def w_map(t, x):
        return w

    def compute_traj(x0, t_end):
        # The embedding system trajectory gives an overapproximation of the reachable set
        return clembsys.compute_trajectory(
            0.0,
            t_end,
            x0,
            (w_map,),
            0.125,
            f_kwargs=immutabledict({"corners": corners, "permutations": permutations}),
            solver=solver,
        )

    t_end = 1.5

    x0s = get_partitions_ut(irx.i2ut(x0), N)
    print(f"Using {len(x0s)} partitions")

    # vmap the compute_traj function over the initial partitions
    # vmapped = jax.jit(jax.vmap(compute_traj, (0, None)), backend=device)
    vmapped = jax.vmap(compute_traj, (0, None))
    vmapped(jnp.zeros_like(x0s), 0.125)  # JIT Compilation Step

    print("Finished setup and compilation.")

    # Runs vmapped on x0s 10 times, reporting the runtimes
    trajs, times = run_times(10, vmapped, x0s, t_end)
    avg_runtime, std_runtime = jnp.mean(times), jnp.std(times)
    print(f"partitions: ${len(x0s)}$, {solver}, {avg_runtime} ± {std_runtime}")

    # diffrax has some timesteps that are inf, so we need to filter them out
    tfinite = jnp.where(jnp.isfinite(trajs.ts[0]))[0]
    for t in tfinite:
        if jnp.isfinite(trajs.ys[:, t, :]).all():
            ut_t = trajs.ys[:, t, :]  # The reachable sets at time t
            boxes_t = [irx.ut2i(box) for box in ut_t]  # Converted to intervals
            draw_iarrays(ax, boxes_t, zorder=2)  # Draw the intervals on ax

    # Monte Carlo Simulations
    def mc_wmap(t, x):
        return jnp.array([0.0])

    for mc_x0 in gen_ics(x0, 100):
        mc_traj = clsys.compute_trajectory(
            0.0, t_end, mc_x0, (mc_wmap,), 0.125, solver=solver
        )
        ax.plot(mc_traj.ys[:, 0], mc_traj.ys[:, 1], color="tab:red", zorder=0)

    # Obstacle
    ax.add_patch(Circle((4, 4), 3 / 1.25, lw=0, fc="salmon", zorder=0))

    ax.set_xlim([-0.5, 8.5])
    ax.set_ylim([-0.5, 8.5])
    ax.set_xlabel("$p_x$", labelpad=3)
    ax.set_ylabel("$p_y$", labelpad=3, rotation="horizontal")
    ax.text(
        0,
        8,
        f"partitions: ${len(x0s)}$, {solver}",
        fontsize=16,
        verticalalignment="top",
    )

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 14})
fig, axs = plt.subplots(2, 3, dpi=100, figsize=[10, 7])
fig.subplots_adjust(
    left=0.05, right=0.975, bottom=0.075, top=0.975, wspace=0.15, hspace=0.25
)
axs = axs.reshape(-1)

for i, N in enumerate(
    [
        (1, method, device),
        (2**4, method, device),
        (3**4, method, device),
        (4**4, method, device),
        (5**4, method, device),
        (6**4, method, device),
    ]
):
    compute_and_plot(axs[i], *N)

fig.savefig("vehicle.pdf")
plt.show()

