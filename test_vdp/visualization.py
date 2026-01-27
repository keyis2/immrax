
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils import rk4, euler, rk45, tsit5

mpl.set_loglevel("warning")


# ---------- Core 2D visualization (x–y) ----------
def visualize_flowpipe_xy(
    times=None, lowers=None, uppers=None, *,
    F_fun=None, X0_box=None, t_end=None, n_samples=0,
    draw_boxes=True, draw_samples=True, print_boxes=False,
    # per-time-slice boxes:
    stride=5, box_facealpha=0.5, box_edgecolor="C0", box_linewidth=0.6,
    # sample trajs:
    sample_alpha=0.7, seed=0, figsize=(7.5,5.5),
):
    """
    Draw a 2D projection of the flowpipe (x vs y).

    times: 1D array of time samples (same length as lowers[0]).
    lowers, uppers: list-like of length 2, each element 1D arrays for x and y bounds.
    segments: optional list of dicts with 't0' and 't1' for coarse per-step boxes.
    """
    if draw_boxes:
        assert len(lowers) == 2 and len(uppers) == 2, "This helper targets 2D state (x,y)."
        times = np.asarray(times)
        x_lo, y_lo = np.asarray(lowers[0]), np.asarray(lowers[1])
        x_up, y_up = np.asarray(uppers[0]), np.asarray(uppers[1])

    fig, ax = plt.subplots(1,1, figsize=figsize)

    # Correct visualization: per-time-slice rectangles (thin “tube”)
    if draw_boxes:
        idxs = range(0, len(times), max(1, int(stride)))
        for i in idxs:
            w = float(x_up[i] - x_lo[i])
            h = float(y_up[i] - y_lo[i])
            if w < 0 or h < 0:
                continue
            if print_boxes:
                print(f"lbox at t={times[i]:.3f}: x=[{x_lo[i]:.4f}, {x_up[i]:.4f}], y=[{y_lo[i]:.4f}, {y_up[i]:.4f}]")
            rect = Rectangle((x_lo[i], y_lo[i]), w, h,
                             facecolor=box_edgecolor, alpha=box_facealpha,
                             edgecolor=box_edgecolor, linewidth=box_linewidth)
            ax.add_patch(rect)

    # Sample trajectories (optional)
    if draw_samples and n_samples > 0 and (F_fun is not None) and (X0_box is not None) and (t_end is not None):
        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, t_end, 2001)
        for _ in range(n_samples):
            x0 = [rng.uniform(X0_box[i][0], X0_box[i][1]) for i in range(2)]
            traj = rk4(F_fun, x0, t_grid)
            ax.plot(traj[:,0], traj[:,1], alpha=sample_alpha, linewidth=1.1)

    ax.set_title("TM Flowpipe in state space (x–y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_name = "flowpipe_xy.png"
    fig.savefig(fig_name)
    print(f"Saved figure to {fig_name}")
    return fig, ax

def visualize_flowpipe_xy_new(
    times=None, lowers=None, uppers=None, *,
    F_fun=None, X0_box=None, t_end=None, n_samples=0, file_name="flowpipe_xy.png",
    solver="rk4",
    draw_boxes=True, draw_samples=True, print_boxes=False, aggregate_partitions=True,
    # per-time-slice boxes:
    stride=5, box_facealpha=0.5, box_edgecolor="C0", box_linewidth=0.6,
    # sample trajs:
    sample_alpha=0.7, seed=0, figsize=(7.5,5.5),
):
    """
    Draw a 2D projection of the flowpipe (x vs y).

    times: 1D array of time samples (same length as lowers[0]).
    lowers, uppers: array-like of shape (N-step, M-split,2) for x and y bounds.
    segments: optional list of dicts with 't0' and 't1' for coarse per-step boxes.
    """
    if draw_boxes:
        assert lowers.ndim == 3 and uppers.ndim == 3, "This helper targets 2D state (x,y)."
        times = np.asarray(times)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    n_partitions_ori = n_partitions = lowers.shape[1]
    if aggregate_partitions:
        # Merge all partitions into one big box per time step
        lowers = np.min(lowers, axis=1, keepdims=True)  # (N,1,2)
        uppers = np.max(uppers, axis=1, keepdims=True)  # (N,1,2)
        n_partitions = 1

    # Correct visualization: per-time-slice rectangles (thin “tube”)
    if draw_boxes:
        idxs = range(0, len(times), max(1, int(stride)))
        for i in idxs:
            for j in range(n_partitions):  # over splits
                x_lo, y_lo = float(lowers[i,j,0]), float(lowers[i,j,1])
                x_up, y_up = float(uppers[i,j,0]), float(uppers[i,j,1])
                w = float(x_up - x_lo)
                h = float(y_up - y_lo)
                if w < 0 or h < 0:
                    continue
                if print_boxes:
                    print(f"lbox at t={times[i]:.3f}: x=[{x_lo:.4f}, {x_up:.4f}], y=[{y_lo:.4f}, {y_up:.4f}]")
                rect = Rectangle((x_lo, y_lo), w, h,
                                facecolor=box_edgecolor, alpha=box_facealpha,
                                edgecolor=box_edgecolor, linewidth=box_linewidth)
                ax.add_patch(rect)

    # Sample trajectories (optional)
    if draw_samples and n_samples > 0 and (F_fun is not None) and (X0_box is not None) and (t_end is not None):
        print(f"Using solver: {solver}")
        solver = eval(solver.lower())
        rng = np.random.default_rng(seed)
        t_grid = np.linspace(0.0, t_end, 2001)
        for _ in range(n_samples):
            x0 = [rng.uniform(X0_box[i][0], X0_box[i][1]) for i in range(2)]
            traj = solver(F_fun, x0, t_grid)
            ax.plot(traj[:,0], traj[:,1], alpha=sample_alpha, linewidth=1.1)

    ax.set_title(f"TM Flowpipe in state space (x–y) (Horizon {times[-1]:.2f}, partitions {n_partitions_ori})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_name)
    print(f"Saved figure to {file_name}")
    return fig, ax

# ---------- Adapter from our FlowState list ----------
def visualize_flowpipe_from_states(states, h, *, F_fun=None, X0_box=None, horizon=None,
                                   n_samples=0, draw_boxes=True, draw_samples=True,
                                   stride=5):
    """
    Convert a list of FlowState snapshots into (times, bounds) and plot x–y rectangles.

    states: list of FlowState (index 0 is initial set at t=0).
    h: fixed step size used in reach().
    """
    import numpy as np
    # times at step starts (t_k = k*h)
    N = len(states) - 1
    times = np.linspace(0.0, N*h, N+1)

    # extract per-step physical boxes for x and y
    x_l = []; x_u = []; y_l = []; y_u = []
    for s in states:
        # each s.box = [t_box, x_box, y_box, ...]; use indices 1 and 2 for x,y
        xb, yb = s.box[1], s.box[2]
        x_l.append(float(xb.lo)); x_u.append(float(xb.hi))
        y_l.append(float(yb.lo)); y_u.append(float(yb.hi))

    lowers = [np.array(x_l), np.array(y_l)]
    uppers = [np.array(x_u), np.array(y_u)]

    t_end = horizon if horizon is not None else N*h

    return visualize_flowpipe_xy(
        times, lowers, uppers,
        F_fun=F_fun, X0_box=X0_box, t_end=t_end, n_samples=n_samples,
        draw_boxes=draw_boxes, draw_samples=draw_samples,
        stride=stride
    )

# ---------- Handy RHS for VdP (for sampling) ----------
def vdp_rhs(x):
    x1, x2 = float(x[0]), float(x[1])
    return [x2, (1.0 - x1*x1)*x2 - x1]
