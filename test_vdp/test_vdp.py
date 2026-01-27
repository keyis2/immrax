#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import immrax as irx
from immrax.utils import get_partitions_ut
from visualization import visualize_flowpipe_xy_new, vdp_rhs

# ---- System definition (no controller, no disturbance) -----------------------
class VanDerPol(irx.System):
    def __init__(self, mu: float = 1.0) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.mu = mu

    def f(self, t: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        mu = self.mu
        return jnp.array([x2, (1.0 - x1**2) * x2 - x1])  # classic μ=1 form


def make_embedding(sys: irx.System, kind: str):
    kind = kind.lower()
    if kind == "natemb":
        return irx.natemb(sys)
    if kind == "jacemb":
        return irx.jacemb(sys)
    if kind == "mjacemb":
        return irx.mjacemb(sys)
    raise ValueError(f"Unknown embedding kind: {kind!r}")


# ---- Plot helper -------------------------------------------------------------
def plot_reachtube(ts: jnp.ndarray, boxes, outfile: str = "vdp_immrax_tube.png"):
    # boxes is a Python list of immrax.intervals over time
    ts_np = jax.device_get(ts)
    xL = jnp.array([b.lower[0] for b in boxes])
    xU = jnp.array([b.upper[0] for b in boxes])
    yL = jnp.array([b.lower[1] for b in boxes])
    yU = jnp.array([b.upper[1] for b in boxes])

    xL, xU, yL, yU = map(jax.device_get, (xL, xU, yL, yU))

    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True, dpi=120)
    axs[0].fill_between(ts_np, xL, xU, alpha=0.35)
    axs[0].set_ylabel(r"$x$")
    axs[0].grid(True, alpha=0.25)

    axs[1].fill_between(ts_np, yL, yU, alpha=0.35)
    axs[1].set_ylabel(r"$y$")
    axs[1].set_xlabel("time (s)")
    axs[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outfile)
    print(f"[saved] {outfile}")

def plot_xy_tube(boxes, every: int = 10, outfile: str = "vdp_immrax_xy.png", alpha: float = 0.10):
    """
    Phase-plot reachtube: draws semi-transparent rectangles for [x,y] boxes
    at decimated time steps, plus the box-centroid path.

    Args:
      boxes: list of immrax Interval objects over time
      every: plot every k-th box to avoid clutter
      outfile: PNG path
      alpha: rectangle transparency
    """
    # Extract arrays
    xL = jnp.array([b.lower[0] for b in boxes])
    xU = jnp.array([b.upper[0] for b in boxes])
    yL = jnp.array([b.lower[1] for b in boxes])
    yU = jnp.array([b.upper[1] for b in boxes])
    xL, xU, yL, yU = map(jax.device_get, (xL, xU, yL, yU))

    xC = 0.5 * (xL + xU)
    yC = 0.5 * (yL + yU)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    # Semi-transparent rectangles for a subset of time steps
    step = max(1, int(every))
    for i in range(0, len(boxes), step):
        w = xU[i] - xL[i]
        h = yU[i] - yL[i]
        rect = Rectangle((xL[i], yL[i]), w, h, linewidth=0, alpha=alpha)
        ax.add_patch(rect)

    # Centerline for visual context
    ax.plot(xC, yC, lw=1.5)

    # Start/end markers
    ax.plot([xC[0]], [yC[0]], marker="o", ms=5, label="start")
    ax.plot([xC[-1]], [yC[-1]], marker="x", ms=6, label="end")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title("Van der Pol phase reachtube (immrax embedding)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile)
    print(f"[saved] {outfile}")

def main():

    # natemb, jacemb, mjacemb
    emb_method = "natemb"
    # euler, tsit5, rk45
    solver = "euler"
    dt = 0.01
    T = 1.0
    mu = 1.0
    n_partitions = 1
    print(f"Embedding: {emb_method}, solver: {solver}, dt={dt}, T={T}, N-partitions={n_partitions}")

    # System and embedding
    sys = VanDerPol(mu=mu)

    emb = make_embedding(sys, emb_method)

    # Initial set: x0 ∈ [1.1, 1.4] × [2.35, 2.45]
    x0 = irx.interval(jnp.array([1.1, 2.35]), jnp.array([1.4, 2.45]))

    # Convert to upper-triangular coordinate expected by EmbeddingSystem
    # x0_ut = irx.i2ut(x0)
    t0 = time.time()
    x0s = get_partitions_ut(irx.i2ut(x0), n_partitions)
    t1 = time.time()
    print(f"Created {len(x0s)} partitions in {t1 - t0:.6f} s")

    # Compute embedding trajectory on [0, T] with step dt
    # (No inputs: pass inputs=())
    compute_traj = lambda ut: emb.compute_trajectory(t0=0.0, tf=T, x0=ut, inputs=(), dt=dt, solver=solver)

    # 1) Warm-up: triggers JIT compile for these exact shapes
    t0 = time.time()
    trajs_warm = jax.vmap(compute_traj)(x0s)
    jax.block_until_ready(trajs_warm.ys)   # force compile + execution to finish
    t1 = time.time()
    print(f"[warm-up] compile+run: {t1 - t0:.6f} s")

    # 2) Timed run: same shapes, so this measures execution only
    t2 = time.time()
    trajs = jax.vmap(compute_traj)(x0s)
    jax.block_until_ready(trajs.ys)        # sync before stopping the timer
    t3 = time.time()
    print(f"[after-JIT] run: {t3 - t2:.6f} s")

    # Convert the (2n)-dim UT state back to Interval at each time
    # (P, N, 2n): vmapped over P partitions
    n_steps = int(T / dt) + 1
    ys_np = np.asarray(jax.device_get(trajs.ys))[:, :n_steps, :]  # (P,N,2n)
    times = np.asarray(jax.device_get(trajs.ts))[0, :n_steps]  # (T,)
    P, N, twoN = ys_np.shape
    n = twoN // 2
    # transpose to (N, P, 2) for visualize_flowpipe_xy_new
    lowers = np.transpose(ys_np[:, :, :n], (1,0,2))   # (N,P,2)
    uppers = np.transpose(ys_np[:, :,  n:], (1,0,2))  # (N,P,2)
    # build per-time HULL boxes for your time-series plot (x(t), y(t))
    hull_lo = lowers.min(axis=1)   # (N,2)
    hull_hi = uppers.max(axis=1)   # (N,2)
    boxes   = [irx.interval(jnp.array(hull_lo[i]), jnp.array(hull_hi[i])) for i in range(N)]
    # X0_box as simple numeric bounds ([(x_lo,x_hi),(y_lo,y_hi)])
    X0_box = [
        (float(x0.lower[0]), float(x0.upper[0])),
        (float(x0.lower[1]), float(x0.upper[1]))
    ]

    # Report final bounds
    final_box = boxes[-1]
    print("[final interval at T={:.3f}]".format(float(T)))
    print("  x(T) ∈ [{:.8f}, {:.8f}]".format(float(final_box.lower[0]), float(final_box.upper[0])))
    print("  y(T) ∈ [{:.8f}, {:.8f}]".format(float(final_box.lower[1]), float(final_box.upper[1])))

    # # Save a time-series reachtube plot
    # plot_reachtube(times, boxes, outfile="vdp_immrax_tube.png")
    # plot_xy_tube(boxes, every=1, outfile="vdp_immrax_xy.png", alpha=0.10)
    visualize_flowpipe_xy_new(
        times=times,
        lowers=lowers,
        uppers=uppers,
        F_fun=vdp_rhs,          # you will define this
        X0_box=X0_box,
        t_end=float(T),
        n_samples=40,
        file_name=f"output/immrax_{emb_method}_{solver}_{T}_{n_partitions}_{t3 - t2:.3f}.png",
        solver=solver,
        print_boxes=False,
        draw_boxes=True,
        draw_samples=True,
        aggregate_partitions=False,
        stride=1
    )

if __name__ == "__main__":
    main()
