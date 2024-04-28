import h5py
import scipy.sparse as sp
import numpy as np


class MeshHyperElasticity3D:
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.filename = f'HyperElasticity3D/mesh_data/HyperElasticity_level{mesh_level}.h5'
        self.load_data(self.filename)
        self.compute_initial_guess()
        self.compute_elastic_nullspace()

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
        coords = self.params["nodes2coord"].ravel()
        self.u_init = coords[self.params["dofsMinim"].ravel()]

    def compute_elastic_nullspace(self):
        coords = self.params["nodes2coord"]

        # Number of nodes
        N = coords.shape[0]

        # Initialize rigid body modes matrix with 6 modes
        rigid_modes = np.zeros((3 * N, 6))

        # Translational modes
        rigid_modes[::3, 0] = 1  # Translation in X
        rigid_modes[1::3, 1] = 1  # Translation in Y
        rigid_modes[2::3, 2] = 1  # Translation in Z

        # Rotational modes about the X, Y, Z axes
        rigid_modes[1::3, 3] = -coords[:, 2]
        rigid_modes[2::3, 3] = coords[:, 1]

        rigid_modes[::3, 4] = coords[:, 2]
        rigid_modes[2::3, 4] = -coords[:, 0]

        rigid_modes[::3, 5] = -coords[:, 1]
        rigid_modes[1::3, 5] = coords[:, 0]

        self.elastic_kernel = rigid_modes[self.params["dofsMinim"].ravel(), :]

    def get_data_jax(self):
        import jax.numpy as jnp
        from jax import config

        config.update("jax_enable_x64", True)

        params = {
            "u_0": jnp.array(self.params["u0"], dtype=jnp.float64),
            "freedofs": jnp.array(self.params["dofsMinim"], dtype=jnp.int32),
            "elems": jnp.array(self.params["elems2nodes"], dtype=jnp.int32),
            "dvx": jnp.array(self.params["dphix"], dtype=jnp.float64),
            "dvy": jnp.array(self.params["dphiy"], dtype=jnp.float64),
            "dvz": jnp.array(self.params["dphiz"], dtype=jnp.float64),
            "vol": jnp.array(self.params["vol"], dtype=jnp.float64),
            "C1": float(self.params["C1"]),
            "D1": float(self.params["D1"]),
        }
        return params, self.adjacency, jnp.array(self.u_init, dtype=jnp.float64)
