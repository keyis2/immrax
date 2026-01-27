import numpy as np

def rk4(F_fun, x0, t_grid):
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    dt = t_grid[1]-t_grid[0]
    for _ in t_grid[:-1]:
        k1 = np.array(F_fun(x))
        k2 = np.array(F_fun(x + 0.5*dt*k1))
        k3 = np.array(F_fun(x + 0.5*dt*k2))
        k4 = np.array(F_fun(x + dt*k3))
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        xs.append(x.copy())
    return np.stack(xs, axis=0)


def euler(F_fun, x0, t_grid):
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    for i in range(len(t_grid) - 1):
        dt = float(t_grid[i+1] - t_grid[i])
        x = x + dt * np.array(F_fun(x))
        xs.append(x.copy())
    return np.stack(xs, axis=0)

# --- generic fixed-step ERK runner -------------------------------------------
def _erk_fixed(F_fun, x0, t_grid, A, b, c=None, use_fsal=False):
    """
    Run a fixed-step explicit Runge–Kutta with tableau (A,b[,c]).
    A: (s,s) strictly lower-triangular (a_ij), b: (s,), c: (s,) stages.
    """
    x = np.array(x0, dtype=float)
    xs = [x.copy()]
    s = len(b)
    k_prev_last = None  # FSAL reuse (k_s from previous step)
    for i in range(len(t_grid) - 1):
        dt = float(t_grid[i+1] - t_grid[i])
        k = [None] * s

        for j in range(s):
            if use_fsal and j == 0 and k_prev_last is not None:
                k[0] = k_prev_last  # FSAL
            else:
                # stage state: x + dt * sum_{m<j} a_{j,m} k[m]
                incr = 0.0
                if j > 0:
                    arow = A[j, :j]
                    incr = dt * sum(arow[m] * k[m] for m in range(j))
                k[j] = np.array(F_fun(x + incr))

        # step update
        x = x + dt * sum(b[j] * k[j] for j in range(s))
        xs.append(x.copy())

        # FSAL cache
        k_prev_last = k[-1]
    return np.stack(xs, axis=0)

# --- Dormand–Prince 5(4) (RK45) coefficients (fixed-step) --------------------
# Wikipedia tableau; 7 stages with FSAL. b is the 5th-order weights. :contentReference[oaicite:2]{index=2}
_A_DP = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
], dtype=float)
_b_DP = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)
_c_DP = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float)

def rk45(F_fun, x0, t_grid):
    return _erk_fixed(F_fun, x0, t_grid, _A_DP, _b_DP, _c_DP, use_fsal=True)

# --- Tsit5 placeholder: same API; swap in Tsitouras 5/4 coefficients here ----
# If/when you want *true* Tsit5, paste its (A,b,c) below and leave use_fsal=True.
# (Tsit5 is also a 7-stage 5(4) ERK with FSAL.) :contentReference[oaicite:3]{index=3}
_A_TSIT = _A_DP  # <-- replace with Tsit5 A
_b_TSIT = _b_DP  # <-- replace with Tsit5 b (5th-order weights)
_c_TSIT = _c_DP  # <-- replace with Tsit5 c

def tsit5(F_fun, x0, t_grid):
    return _erk_fixed(F_fun, x0, t_grid, _A_TSIT, _b_TSIT, _c_TSIT, use_fsal=True)

