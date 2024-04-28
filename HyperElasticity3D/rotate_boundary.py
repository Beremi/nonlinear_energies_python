import jax.numpy as jnp


def rotate_boundary(params, angle):
    u0 = params["u_0"].reshape(-1, 3)
    u0_new = u0.flatten()
    lx = jnp.max(u0[:, 0])
    nodes = jnp.where(u0[:, 0] == lx)[0]

    u0_new = u0_new.at[nodes * 3 + 1].set(jnp.cos(angle) * u0[nodes, 1] +
                                          jnp.sin(angle) * u0[nodes, 2])
    u0_new = u0_new.at[nodes * 3 + 2].set(-jnp.sin(angle) * u0[nodes, 1] +
                                          jnp.cos(angle) * u0[nodes, 2])

    params["u_0"] = u0_new
    return params
