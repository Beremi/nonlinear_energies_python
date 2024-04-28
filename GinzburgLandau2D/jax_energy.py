import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def J(u, u_0, freedofs, elems, dvx, dvy, vol, ip, w, eps):
    v = u_0.at[freedofs].set(u)
    v_elems = v[elems]

    F_x = jnp.sum(v_elems * dvx, axis=1)
    F_y = jnp.sum(v_elems * dvy, axis=1)

    e_1 = (1 / 2) * eps * (F_x**2 + F_y**2)
    e_2 = (1 / 4) * ((v_elems @ ip)**2 - 1)**2 @ w
    return jnp.sum((e_1 + e_2) * vol)
