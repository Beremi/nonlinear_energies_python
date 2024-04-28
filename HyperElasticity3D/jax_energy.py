import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def energy_jax(u, u0, dofsMinim, elems2nodes, dphix, dphiy, dphiz, vol, C1, D1):
    v = u0.at[dofsMinim].set(u)
    vx = v[0::3][elems2nodes]
    vy = v[1::3][elems2nodes]
    vz = v[2::3][elems2nodes]

    G11 = jnp.sum(vx * dphix, axis=1)
    G12 = jnp.sum(vx * dphiy, axis=1)
    G13 = jnp.sum(vx * dphiz, axis=1)
    G21 = jnp.sum(vy * dphix, axis=1)
    G22 = jnp.sum(vy * dphiy, axis=1)
    G23 = jnp.sum(vy * dphiz, axis=1)
    G31 = jnp.sum(vz * dphix, axis=1)
    G32 = jnp.sum(vz * dphiy, axis=1)
    G33 = jnp.sum(vz * dphiz, axis=1)

    I1 = (G11**2 + G12**2 + G13**2 + G21**2 + G22**2 +
          G23**2 + G31**2 + G32**2 + G33**2)
    det = jnp.abs(G11 * G22 * G33 - G11 * G23 * G32 -
                  G12 * G21 * G33 + G12 * G23 * G31 +
                  G13 * G21 * G32 - G13 * G22 * G31)
    W = C1 * (I1 - 3 - 2 * jnp.log(det)) + D1 * (det - 1)**2
    return jnp.sum(W * vol)
