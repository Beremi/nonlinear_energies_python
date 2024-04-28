import time
from functools import partial

import jax
import numpy as np
import scipy.sparse as sps
from jax import config

from .graph_sfd import color_connectivity_for_adjacency, coloring_to_grouping

config.update("jax_enable_x64", True)


class EnergyDerivator:
    def __init__(self, energy_jax_fun, params, adjacency, u_init):
        self.energy_jax_fun = energy_jax_fun
        self.params = params
        self.adjacency = adjacency
        self.u_init = u_init
        self.timings = {}
        self.prepare_sfd()
        self.compile_derivatives()

    def prepare_sfd(self) -> None:
        coloring_start = time.perf_counter()
        self.coloring, self.compressed_index, self.sparse_index = color_connectivity_for_adjacency(self.adjacency)
        self.grouping = coloring_to_grouping(self.coloring)
        coloring_stop = time.perf_counter()
        self.timings['coloring'] = coloring_stop - coloring_start

    def compile_derivatives(self) -> None:
        compile_start = time.perf_counter()
        self.f = jax.jit(self.energy_jax_fun)
        _ = self.f(self.u_init, **self.params)

        self.df_raw = jax.grad(self.energy_jax_fun, argnums=0)
        self.df = jax.jit(self.df_raw)
        _ = self.df(self.u_init, **self.params)

        def ddf_tangent_raw(x, tangent, params):
            def ddf_inner(x):
                return self.df(x, **params)
            return jax.jvp(ddf_inner, (x,), (tangent,))[1]

        self.ddf_tangent_raw = ddf_tangent_raw
        self.ddf_tangent = jax.jit(ddf_tangent_raw)
        _ = self.ddf_tangent(self.u_init, self.u_init, self.params)

        def ddf_data_block(u, grouping, params):
            n_batch_data = grouping.shape[1]
            result = []
            for i in range(0, n_batch_data):
                result.append(self.ddf_tangent(u, grouping[:, i], params))
            return np.vstack(result).T
        self.ddf_data_block = ddf_data_block
        compile_stop = time.perf_counter()
        self.timings['compilation'] = compile_stop - compile_start

    def get_derivatives(self):
        def f_numpy(u, params):
            return np.array(self.f(u, **params))

        def df_numpy(u, params):
            return np.array(self.df(u, **params))

        f = partial(f_numpy, params=self.params)
        df = partial(df_numpy, params=self.params)
        hess_shape = self.adjacency.shape

        def ddf_sparse(u, grouping, compressed_index, sparse_index, hess_shape, params):
            data_hess_block = self.ddf_data_block(u, grouping, params)
            data = data_hess_block[compressed_index]
            H = sps.csr_matrix((data, sparse_index), shape=hess_shape)
            H.eliminate_zeros()
            return H

        ddf = partial(ddf_sparse, grouping=self.grouping, compressed_index=self.compressed_index,
                      sparse_index=self.sparse_index, hess_shape=hess_shape, params=self.params)

        return f, df, ddf
