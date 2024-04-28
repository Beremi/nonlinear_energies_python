import h5py
import scipy.sparse as sp
import numpy as np


class MeshpLaplace2D:
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.filename = f'pLaplace2D/mesh_data/pLaplace_level{mesh_level}.h5'
        self.load_data(self.filename)
        self.compute_initial_guess()

    def load_data(self, filename):
        self.params = {}
        with h5py.File(filename, 'r') as file:
            for key in file:
                if key == 'adjacency':
                    # Handle the COO sparse matrix separately
                    grp = file[key]
                    data = grp['data'][:]
                    row = grp['row'][:]
                    col = grp['col'][:]
                    shape = tuple(grp['shape'][:])
                    # Reconstruct the COO matrix
                    self.adjacency = sp.coo_matrix((data, (row, col)), shape=shape)
                else:
                    # Check if the dataset is a scalar
                    if file[key].shape == ():  # This checks if the dataset is a scalar
                        self.params[key] = file[key][()]  # Retrieve the scalar value directly
                    else:
                        # Load array datasets directly
                        self.params[key] = file[key][:]

    def compute_initial_guess(self):
        np.random.seed(0)
        self.u_init = np.random.rand(self.params["freedofs"].size)

    def get_data_jax(self):
        import jax.numpy as jnp
        from jax import config

        config.update("jax_enable_x64", True)

        params = {
            "u_0": jnp.array(self.params["u_0"], dtype=jnp.float64),
            "freedofs": jnp.array(self.params["freedofs"], dtype=jnp.int32),
            "elems": jnp.array(self.params["elems"], dtype=jnp.int32),
            "dvx": jnp.array(self.params["dvx"], dtype=jnp.float64),
            "dvy": jnp.array(self.params["dvy"], dtype=jnp.float64),
            "vol": jnp.array(self.params["vol"], dtype=jnp.float64),
            "p": float(self.params["p"]),
            "f": jnp.array(self.params["f"], dtype=jnp.float64)
        }
        return params, self.adjacency, jnp.array(self.u_init, dtype=jnp.float64)
