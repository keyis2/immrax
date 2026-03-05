import jax
import jax.numpy as jnp
import immrax as ir

# -----------------------
# 1) Define function
# -----------------------
def f(x):
    x1, x2 = x
    return jnp.array([jnp.sin(x1) + x1 * x2])

# -----------------------
# 2) Define box X
# -----------------------
X = ir.Interval(
    lower=jnp.array([0.9, 1.9]),
    upper=jnp.array([1.1, 2.1])
)

# Center x'
x_center = jnp.array([1.0, 2.0])

# -----------------------
# 3) Natural inclusion (optional)
# -----------------------
f_nat = ir.natif(f)
print("Natural inclusion f(X):", f_nat(X))

# -----------------------
# 4) Interval Jacobian enclosure
# -----------------------
jac_inclusion = ir.natif(jax.jacfwd(f))
J_interval = jac_inclusion(X)
print("\nInterval Jacobian [J]:")
print(J_interval)

# Jacobian-based inclusion bound
jac_if = ir.jacif(f)
print("\nJacobian-based inclusion f(X):")
print(jac_if(X))

# -----------------------
# 5) Mixed Jacobian enclosure
# -----------------------
mjac_matrix = ir.mjacM(f)

M_interval = mjac_matrix(X)
print("\nMixed Jacobian interval [M]:")
print(M_interval)

# Mixed Jacobian-based inclusion bound
mjac_if = ir.mjacif(f)

print("\nMixed Jacobian-based inclusion f(X):")
print(mjac_if(X))