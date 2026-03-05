import jax
# jax.config.update("jax_platform_name", "cpu")  # for testing on GPU/TPU machines without device arrays
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import immrax as irx
import time

def interval_mul_lo_up(a_lo, a_up, b_lo, b_up):
    p1 = a_lo * b_lo
    p2 = a_lo * b_up
    p3 = a_up * b_lo
    p4 = a_up * b_up
    lo = jnp.minimum(jnp.minimum(p1, p2), jnp.minimum(p3, p4))
    up = jnp.maximum(jnp.maximum(p1, p2), jnp.maximum(p3, p4))
    return lo, up

def make_slice_box_midpoint(X: irx.Interval, ell: int):
    # ell is number of free dims (0..n). ell=0 means all fixed -> point at midpoint
    lo, up = X.lower, X.upper
    x0 = 0.5 * (lo + up)
    lo2 = lo.at[ell:].set(x0[ell:])
    up2 = up.at[ell:].set(x0[ell:])
    return irx.interval(lo2, up2), x0

def remainder_from_H_interval(H_lo, H_up, d_lo, d_up, cross_mask):
    """
    H_lo/H_up: (n,n); d_lo/d_up: (n,)
    cross_mask: (n,n) boolean mask selecting j<k entries
    Returns scalar interval (lo,up)
    """
    # delta^2 intervals (elementwise)
    d2_lo, d2_up = interval_mul_lo_up(d_lo, d_up, d_lo, d_up)  # (n,)

    # diagonal contribution: 0.5 * sum_k H_kk * d_k^2
    Hdiag_lo = jnp.diag(H_lo)
    Hdiag_up = jnp.diag(H_up)
    diag_lo, diag_up = interval_mul_lo_up(Hdiag_lo, Hdiag_up, d2_lo, d2_up)
    diag_lo = 0.5 * diag_lo
    diag_up = 0.5 * diag_up

    # cross contribution: sum_{j<k} H_jk * (d_j d_k)
    # build (n,n) interval for d_j d_k via outer products with broadcasting
    dj_lo = d_lo[:, None]
    dj_up = d_up[:, None]
    dk_lo = d_lo[None, :]
    dk_up = d_up[None, :]
    dd_lo, dd_up = interval_mul_lo_up(dj_lo, dj_up, dk_lo, dk_up)  # (n,n)

    cross_lo, cross_up = interval_mul_lo_up(H_lo, H_up, dd_lo, dd_up)  # (n,n)

    # keep only j<k
    cross_lo = jnp.where(cross_mask, cross_lo, 0.0)
    cross_up = jnp.where(cross_mask, cross_up, 0.0)

    lo = jnp.sum(diag_lo) + jnp.sum(cross_lo)
    up = jnp.sum(diag_up) + jnp.sum(cross_up)
    return lo, up

def remainder_bound_classic_hessian(f, x_lo, x_up):
    x0 = 0.5 * (x_lo + x_up)                      # midpoint nominal
    d_lo, d_up = x_lo - x0, x_up - x0             # delta interval (n,)
    n = x_lo.shape[0]

    # Hessian interval for vector-output f: (m,n,n)
    X = irx.interval(x_lo, x_up)
    H = irx.natif(jax.hessian(f))(X)
    H_lo, H_up = H.lower, H.upper
    m = H_lo.shape[0]

    # delta^2 interval (n,)
    d2_lo, d2_up = interval_mul_lo_up(d_lo, d_up, d_lo, d_up)

    # diagonal contribution: 0.5 * sum_k H[:,k,k] * d2[k]
    Hdiag_lo = jnp.diagonal(H_lo, axis1=1, axis2=2)  # (m,n)
    Hdiag_up = jnp.diagonal(H_up, axis1=1, axis2=2)  # (m,n)
    diag_lo, diag_up = interval_mul_lo_up(Hdiag_lo, Hdiag_up, d2_lo[None, :], d2_up[None, :])
    diag_lo = 0.5 * jnp.sum(diag_lo, axis=1)         # (m,)
    diag_up = 0.5 * jnp.sum(diag_up, axis=1)         # (m,)

    # cross contribution: sum_{j<k} H[:,j,k] * (d[j] d[k])
    # build dd interval (n,n)
    dj_lo, dj_up = d_lo[:, None], d_up[:, None]
    dk_lo, dk_up = d_lo[None, :], d_up[None, :]
    dd_lo, dd_up = interval_mul_lo_up(dj_lo, dj_up, dk_lo, dk_up)  # (n,n)

    cross_lo, cross_up = interval_mul_lo_up(H_lo, H_up, dd_lo[None, :, :], dd_up[None, :, :])  # (m,n,n)
    mask = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    cross_lo = jnp.where(mask[None, :, :], cross_lo, 0.0)
    cross_up = jnp.where(mask[None, :, :], cross_up, 0.0)
    cross_lo = jnp.sum(cross_lo, axis=(1, 2))  # (m,)
    cross_up = jnp.sum(cross_up, axis=(1, 2))  # (m,)

    r_lo = diag_lo + cross_lo
    r_up = diag_up + cross_up
    return irx.interval(r_lo, r_up)

def interval_add_lo_up(a_lo, a_up, b_lo, b_up):
    return a_lo + b_lo, a_up + b_up

def interval_matvec_lo_up(A_lo, A_up, x_lo, x_up):
    """
    Interval matrix-vector product y = A x.
    A_lo/A_up: (..., m, n)
    x_lo/x_up: (..., n)
    Returns y_lo/y_up: (..., m)
    """
    # Expand x for broadcasting over matrix columns
    xlo = x_lo[..., None, :]  # (...,1,n)
    xup = x_up[..., None, :]

    # Elementwise interval product for each entry A_ij * x_j
    prod_lo, prod_up = interval_mul_lo_up(A_lo, A_up, xlo, xup)  # (...,m,n)

    # Sum across columns
    y_lo = jnp.sum(prod_lo, axis=-1)
    y_up = jnp.sum(prod_up, axis=-1)
    return y_lo, y_up

def interval_dot_lo_up(a_lo, a_up, b_lo, b_up):
    """
    Interval dot product s = a^T b.
    a_lo/a_up, b_lo/b_up: (..., n)
    Returns scalar interval (...,)
    """
    prod_lo, prod_up = interval_mul_lo_up(a_lo, a_up, b_lo, b_up)  # (...,n)
    s_lo = jnp.sum(prod_lo, axis=-1)
    s_up = jnp.sum(prod_up, axis=-1)
    return s_lo, s_up

def remainder_bound_classic_hessian_matrixstyle(f, x_lo, x_up):
    """
    Classic interval Hessian remainder bound, but compute quadratic form as:
        r2_i(X) ⊆ 0.5 * δ^T [H_i(X)] δ
    using interval matvec + interval dot (no explicit diagonal/cross expansion).

    Returns Interval shape (m,).
    """
    x0 = 0.5 * (x_lo + x_up)
    d_lo, d_up = x_lo - x0, x_up - x0
    n = x_lo.shape[0]

    # Vector-output Hessian inclusion: (m,n,n)
    X = irx.interval(x_lo, x_up)
    H = irx.natif(jax.hessian(f))(X)
    H_lo, H_up = H.lower, H.upper
    m = H_lo.shape[0]

    # Compute y = [H] δ  (interval), y shape (m,n)
    # Treat each output i as its own (n,n) matrix: batch over i
    y_lo, y_up = interval_matvec_lo_up(H_lo, H_up, d_lo[None, :], d_up[None, :])  # (m,n)

    # Compute s = δ^T y  (interval), s shape (m,)
    s_lo, s_up = interval_dot_lo_up(d_lo[None, :], d_up[None, :], y_lo, y_up)     # (m,)

    # half factor
    r_lo = 0.5 * s_lo
    r_up = 0.5 * s_up
    return irx.interval(r_lo, r_up)

def remainder_bound_path_based(f, x_lo, x_up):
    """
    Path-based remainder bound (identity order), midpoint nominal.
    Only loops over slices (n calls). Fully vectorized over outputs and (j,k) pairs.
    Returns Interval shape (m,).
    """
    x0 = 0.5 * (x_lo + x_up)
    d_lo, d_up = x_lo - x0, x_up - x0
    n = x_lo.shape[0]

    # Build slices X^(1)..X^(n) (python list)
    X = irx.interval(x_lo, x_up)
    X_slices = [make_slice_box_midpoint(X, ell)[0] for ell in range(1, n + 1)]

    # For each slice, compute vector-output Hessian interval: (m,n,n)
    # NOTE: this is the only loop we keep
    Hlo_list = []
    Hup_list = []
    hess_incl = irx.natif(jax.hessian(f))   # inclusion of Hessian operator
    for Xell in X_slices:
        H = hess_incl(Xell)                # H.lower shape (m,n,n)
        Hlo_list.append(H.lower)
        Hup_list.append(H.upper)

    # Stack slices: (n,m,n,n)
    Hb_lo = jnp.stack(Hlo_list, axis=0)
    Hb_up = jnp.stack(Hup_list, axis=0)
    m = Hb_lo.shape[1]

    # delta^2 (n,)
    d2_lo, d2_up = interval_mul_lo_up(d_lo, d_up, d_lo, d_up)

    # ---- diagonal terms: slice k uses X^(k+1) => slice index k
    # Hb_lo[k, :, k, k] gives (n,m)
    kk = jnp.arange(n)
    Hkk_lo = Hb_lo[kk, :, kk, kk]  # (n,m)
    Hkk_up = Hb_up[kk, :, kk, kk]  # (n,m)
    diag_lo, diag_up = interval_mul_lo_up(Hkk_lo, Hkk_up, d2_lo[:, None], d2_up[:, None])  # (n,m)
    diag_lo = 0.5 * jnp.sum(diag_lo, axis=0)  # (m,)
    diag_up = 0.5 * jnp.sum(diag_up, axis=0)  # (m,)

    # ---- cross terms: for each (j,k), j<k, use slice j (X^(j+1) => slice index j)
    j_idx, k_idx = jnp.triu_indices(n, k=1)  # (P,)
    P = j_idx.shape[0]

    # H_jk interval per pair: Hb_lo[j, :, j, k] => (P,m)
    Hjk_lo = Hb_lo[j_idx, :, j_idx, k_idx]  # (P,m)
    Hjk_up = Hb_up[j_idx, :, j_idx, k_idx]  # (P,m)

    # delta_j delta_k interval per pair: (P,)
    dj_lo, dj_up = d_lo[j_idx], d_up[j_idx]
    dk_lo, dk_up = d_lo[k_idx], d_up[k_idx]
    djdk_lo, djdk_up = interval_mul_lo_up(dj_lo, dj_up, dk_lo, dk_up)  # (P,)

    cross_lo, cross_up = interval_mul_lo_up(Hjk_lo, Hjk_up, djdk_lo[:, None], djdk_up[:, None])  # (P,m)
    cross_lo = jnp.sum(cross_lo, axis=0)  # (m,)
    cross_up = jnp.sum(cross_up, axis=0)  # (m,)

    r_lo = diag_lo + cross_lo
    r_up = diag_up + cross_up
    return irx.interval(r_lo, r_up)

def remainder_bound_path_based(f, x_lo, x_up):
    x0 = 0.5 * (x_lo + x_up)
    d_lo, d_up = x_lo - x0, x_up - x0
    n = x_lo.shape[0]

    # Build batched slice boxes: batch b corresponds to ell=b+1 free dims
    # free_mask[b, i] = True iff i <= b  (i < b+1)
    idx = jnp.arange(n)[None, :]            # (1,n)
    ell = jnp.arange(1, n + 1)[:, None]     # (n,1)
    free_mask = (idx < ell)                  # (n,n)

    lo_b = jnp.where(free_mask, x_lo[None, :], x0[None, :])  # (n,n)
    up_b = jnp.where(free_mask, x_up[None, :], x0[None, :])  # (n,n)
    Xb = irx.interval(lo_b, up_b)

    def int_hess_slice(lo, up):
        H = irx.natif(jax.hessian(f))(irx.interval(lo, up))
        return H.lower, H.upper

    # Batched Hessian interval: (n,m,n,n)
    Hb_lo, Hb_up = jax.vmap(int_hess_slice)(lo_b, up_b)  # (n,m,n,n)
    m = Hb_lo.shape[1]

    # delta^2 for diagonal
    d2_lo, d2_up = interval_mul_lo_up(d_lo, d_up, d_lo, d_up)  # (n,)

    # ----- diagonal: use slice k (batch k), entry (k,k)
    # pick Hb[k, :, k, k] -> (n,m)
    Hkk_lo = Hb_lo[jnp.arange(n), :, jnp.arange(n), jnp.arange(n)]  # (n,m)
    Hkk_up = Hb_up[jnp.arange(n), :, jnp.arange(n), jnp.arange(n)]  # (n,m)

    diag_lo, diag_up = interval_mul_lo_up(
        Hkk_lo, Hkk_up,
        d2_lo[:, None], d2_up[:, None]
    )  # (n,m)
    diag_lo = 0.5 * jnp.sum(diag_lo, axis=0)  # (m,)
    diag_up = 0.5 * jnp.sum(diag_up, axis=0)  # (m,)

    # ----- cross: for each (j,k) with j<k, use slice j (batch j), entry (j,k)
    j_idx, k_idx = jnp.triu_indices(n, k=1)   # (P,), (P,)
    # coeff interval per pair: Hb[j, :, j, k] -> (P,m)
    Hjk_lo = Hb_lo[j_idx, :, j_idx, k_idx]    # (P,m)
    Hjk_up = Hb_up[j_idx, :, j_idx, k_idx]    # (P,m)

    # delta_j delta_k interval per pair: (P,)
    dj_lo, dj_up = d_lo[j_idx], d_up[j_idx]
    dk_lo, dk_up = d_lo[k_idx], d_up[k_idx]
    djdk_lo, djdk_up = interval_mul_lo_up(dj_lo, dj_up, dk_lo, dk_up)  # (P,)

    cross_lo, cross_up = interval_mul_lo_up(
        Hjk_lo, Hjk_up,
        djdk_lo[:, None], djdk_up[:, None]
    )  # (P,m)
    cross_lo = jnp.sum(cross_lo, axis=0)  # (m,)
    cross_up = jnp.sum(cross_up, axis=0)  # (m,)

    r_lo = diag_lo + cross_lo
    r_up = diag_up + cross_up
    return irx.interval(r_lo, r_up)

def satellite_ct_dynamics(x):
    # unpack state and input
    q0, q1, q2, q3, w1, w2, w3, u1, u2, u3 = [x[i] for i in range(10)]

    # --------------------------
    # Quaternion dynamics q_dot = Omega(w) q (expanded dim-wise)
    # Omega(w) = 0.5 * [[0,-w1,-w2,-w3],
    #                   [w1,0,w3,-w2],
    #                   [w2,-w3,0,w1],
    #                   [w3,w2,-w1,0]]
    # --------------------------
    dq0 = 0.5 * (      - w1*q1 - w2*q2 - w3*q3)
    dq1 = 0.5 * (w1*q0         + w3*q2 - w2*q3)
    dq2 = 0.5 * (w2*q0 - w3*q1         + w1*q3)
    dq3 = 0.5 * (w3*q0 + w2*q1 - w1*q2        )

    # --------------------------
    # Angular-rate dynamics:
    # w_dot = I^{-1}(v - w x (I w)), I = diag(5,2,1)
    # Iw = [5w1, 2w2, 1w3]
    # w x (Iw) =
    #   [ w2*w3*(1-2),  w1*w3*(5-1),  w1*w2*(2-5) ]
    # = [ -w2*w3,       4*w1*w3,      -3*w1*w2 ]
    #
    # Therefore:
    # dw1 = (u1 - (-w2*w3))/5 = (u1 + w2*w3)/5
    # dw2 = (u2 - 4*w1*w3)/2
    # dw3 = (u3 - (-3*w1*w2))/1 = u3 + 3*w1*w2
    # --------------------------
    dw1 = (u1 + w2 * w3) / 5.0
    dw2 = (u2 - 4.0 * w1 * w3) / 2.0
    dw3 = (u3 + 3.0 * w1 * w2)

    du1 = jnp.zeros_like(u1)
    du2 = jnp.zeros_like(u2)
    du3 = jnp.zeros_like(u3)

    return jnp.stack([dq0, dq1, dq2, dq3, dw1, dw2, dw3, du1, du2, du3], axis=0)


def rk4_step_ct(f_ct, x, h=1.0):
    """One RK4 step for x_dot = f_ct(x), with constant input over the step."""
    k1 = f_ct(x)
    k2 = f_ct(x + 0.5 * h * k1)
    k3 = f_ct(x + 0.5 * h * k2)
    k4 = f_ct(x + h * k3)
    x_next = x + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return x_next


def satellite_dt_dynamics(x, h=1.0):
    """
    Discrete-time dynamics from CT via RK4.
    Paper uses h = 1.0.
    """
    x_next = rk4_step_ct(satellite_ct_dynamics, x, h=h)
    return x_next


def euler_zyx_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles (yaw-pitch-roll) to quaternion [q0,q1,q2,q3]
    with q0 = scalar part.
    """
    cr = jnp.cos(roll * 0.5)
    sr = jnp.sin(roll * 0.5)
    cp = jnp.cos(pitch * 0.5)
    sp = jnp.sin(pitch * 0.5)
    cy = jnp.cos(yaw * 0.5)
    sy = jnp.sin(yaw * 0.5)

    # q = [w, x, y, z]
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return jnp.stack([q0, q1, q2, q3], axis=0)


def get_initial_state():
    # Paper initial Euler angles and angular rate (degrees -> radians)
    euler_deg = jnp.array([180.0, 45.0, 45.0])   # (roll, pitch, yaw) if using this convention
    euler_rad = euler_deg * jnp.pi / 180.0

    omega0 = jnp.array([-1.0, -4.5, 4.5]) * jnp.pi / 180.0  # rad/s

    # Initial quaternion from Euler angles
    q0 = euler_zyx_to_quat(euler_rad[0], euler_rad[1], euler_rad[2])

    # Initial state z0 = [q0,q1,q2,q3,w1,w2,w3]
    z0 = jnp.concatenate([q0, omega0], axis=0)

    return z0

def main():
    # Test at a random state
    state_dim = 7
    input_dim = 3
    z_eps = 0.05
    v_eps = 0.1

    dynamics = satellite_dt_dynamics

    x0 = jnp.concatenate([get_initial_state(), jnp.zeros(input_dim)], axis=0)  # initial state + zero input
    eps = jnp.concatenate([jnp.full(state_dim, z_eps), jnp.full(input_dim, v_eps)], axis=0)  # shape (10,)
    x0_lo = (x0 - eps)
    x0_hi = (x0 + eps)
    print(f"x0_lo: {x0_lo}")
    print(f"x0_hi: {x0_hi}")

    # ------------------------------------------------------------
    # Nominal point = midpoint of interval (as we agreed)
    # ------------------------------------------------------------
    x_nom = 0.5 * (x0_lo + x0_hi)

    # Jacobian at nominal
    J_nom = jax.jacfwd(dynamics)(x_nom)  # (10,10)

    # Remainder function r(x) = f(x) - f(x_nom) - J_nom (x-x_nom)
    f_nom = dynamics(x_nom)

    def remainder(x):
        return dynamics(x) - f_nom - J_nom @ (x - x_nom)

    # ------------------------------------------------------------
    # Compute interval remainder bounds (assumes you pasted these in the file)
    #   - remainder_bound_classic_hessian(dynamics, X)
    #   - remainder_bound_path_based(dynamics, X)
    # ------------------------------------------------------------
    # print("\nComputing classic interval-Hessian remainder bound (matrix)...")
    # start_time = time.time()
    # R_classic_matrix = remainder_bound_classic_hessian_matrixstyle(dynamics, x0_lo, x0_hi)
    # end_time = time.time()
    # print(f"Classic Hessian remainder bound (matrix) computed in {end_time - start_time:.6f} seconds")
    # print("R_classic_matrix.lower:", R_classic_matrix.lower)
    # print("R_classic_matrix.upper:", R_classic_matrix.upper)

    print("\nComputing classic interval-Hessian remainder bound...")
    @jax.jit
    def jit_classic(x_lo, x_up):
        return remainder_bound_classic_hessian(dynamics, x_lo, x_up)
    start_time = time.time()
    R_classic = jit_classic(x0_lo, x0_hi)
    jax.block_until_ready(R_classic)
    end_time = time.time()
    print(f"Compile time: {end_time - start_time:.6f} seconds")
    start_time = time.time()
    R_classic = jit_classic(x0_lo, x0_hi)
    jax.block_until_ready(R_classic)
    end_time = time.time()
    print(f"Running time: {(end_time - start_time) * 1000:.6f} ms")
    print("R_classic.lower:", R_classic.lower)
    print("R_classic.upper:", R_classic.upper)

    print("\nComputing path-based remainder bound...")
    @jax.jit
    def jit_path(x_lo, x_up):
        return remainder_bound_path_based(dynamics, x_lo, x_up)
    start_time = time.time()
    R_path = jit_path(x0_lo, x0_hi)
    jax.block_until_ready(R_path)
    end_time = time.time()
    print(f"Compile time: {end_time - start_time:.6f} seconds")
    start_time = time.time()
    R_path = jit_path(x0_lo, x0_hi)
    jax.block_until_ready(R_path)
    end_time = time.time()
    print(f"Running time: {(end_time - start_time) * 1000:.6f} ms")
    print("R_path.lower:", R_path.lower)
    print("R_path.upper:", R_path.upper)

    # # quick width comparison
    # w_classic = R_classic.upper - R_classic.lower
    # w_path = R_path.upper - R_path.lower
    # # print("\nWidth classic (per-dim):", w_classic)
    # # print("Width path    (per-dim):", w_path)
    # # print("Mean width classic:", jnp.mean(w_classic), " path:", jnp.mean(w_path))

    # # ------------------------------------------------------------
    # # Monte Carlo check: sample points in the box, check containment
    # # ------------------------------------------------------------
    # key = jax.random.PRNGKey(0)
    # num_samples = 5000

    # u = jax.random.uniform(key, shape=(num_samples, x_nom.shape[0]))
    # xs = x0_lo[None, :] + u * (x0_hi - x0_lo)[None, :]  # uniform in box

    # rs = jax.vmap(remainder)(xs)  # (N,10)

    # r_min = jnp.min(rs, axis=0)
    # r_max = jnp.max(rs, axis=0)
    # print("\nMC remainder min:", r_min)
    # print("MC remainder max:", r_max)

    # def check_contains(R, name):
    #     ok_lo = jnp.all(r_min >= R.lower - 1e-9)
    #     ok_up = jnp.all(r_max <= R.upper + 1e-9)
    #     print(f"\n[{name}] contains MC min/max? lo_ok={bool(ok_lo)} up_ok={bool(ok_up)}")
    #     # if false, print which dims violated
    #     if not bool(ok_lo):
    #         bad = jnp.where(r_min < R.lower - 1e-9)[0]
    #         print(f"  dims with r_min < lower: {bad.tolist()}")
    #         print("  r_min[bad]:", r_min[bad])
    #         print("  lower[bad]:", R.lower[bad])
    #     if not bool(ok_up):
    #         bad = jnp.where(r_max > R.upper + 1e-9)[0]
    #         print(f"  dims with r_max > upper: {bad.tolist()}")
    #         print("  r_max[bad]:", r_max[bad])
    #         print("  upper[bad]:", R.upper[bad])

    # check_contains(R_classic, "classic")
    # check_contains(R_path, "path-based")

    # # optional: report which one is tighter dimension-wise
    # tighter = (w_path <= w_classic + 1e-12)
    # print("\nPath tighter-or-equal dims:", jnp.where(tighter)[0].tolist())
    # print("Path looser dims:", jnp.where(~tighter)[0].tolist())


if __name__ == "__main__":
    main()
