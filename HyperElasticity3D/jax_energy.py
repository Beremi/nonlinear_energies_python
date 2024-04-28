import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def J(u, u_0, freedofs, elems, dvx, dvy, dvz, vol, C1, D1):
    v = u_0.at[freedofs].set(u)
    vx_elem = v[0::3][elems]
    vy_elem = v[1::3][elems]
    vz_elem = v[2::3][elems]

    F11 = jnp.sum(vx_elem * dvx, axis=1)
    F12 = jnp.sum(vx_elem * dvy, axis=1)
    F13 = jnp.sum(vx_elem * dvz, axis=1)
    F21 = jnp.sum(vy_elem * dvx, axis=1)
    F22 = jnp.sum(vy_elem * dvy, axis=1)
    F23 = jnp.sum(vy_elem * dvz, axis=1)
    F31 = jnp.sum(vz_elem * dvx, axis=1)
    F32 = jnp.sum(vz_elem * dvy, axis=1)
    F33 = jnp.sum(vz_elem * dvz, axis=1)

    I1 = (F11**2 + F12**2 + F13**2 +
          F21**2 + F22**2 + F23**2 +
          F31**2 + F32**2 + F33**2)
    det = jnp.abs(+ F11 * F22 * F33 - F11 * F23 * F32
                  - F12 * F21 * F33 + F12 * F23 * F31
                  + F13 * F21 * F32 - F13 * F22 * F31)
    W = C1 * (I1 - 3 - 2 * jnp.log(det)) + D1 * (det - 1)**2
    return jnp.sum(W * vol)
