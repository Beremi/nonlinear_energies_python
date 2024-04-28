import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def J(u, u_0, freedofs, elems, dvx, dvy, vol, p, f):
    v = u_0.at[freedofs].set(u)
    v_elems = v[elems]

    F_x = jnp.sum(v_elems * dvx, axis=1)
    F_y = jnp.sum(v_elems * dvy, axis=1)

    intgrds = (1 / p) * (F_x**2 + F_y**2)**(p / 2)
    return jnp.sum(intgrds * vol) - jnp.dot(f, v)
