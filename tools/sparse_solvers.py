import pyamg
import scipy.sparse.linalg as spla
from typing import Literal


class SolverAMGElasticity:
    def __init__(self, H, elastic_kernel=None, verbose=False, tol=1e-3, maxiter=1000):
        self.H = H
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.U = elastic_kernel
        ml = pyamg.smoothed_aggregation_solver(self.H.tocsr(), B=self.U, smooth='energy')
        self.M_lin = ml.aspreconditioner()

    def solve(self, x):
        # conjugate gradient solution of self.H with  x
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1

        sol = spla.cg(self.H, x.copy(), rtol=self.tol, M=self.M_lin, callback=callback, maxiter=self.maxiter)[0]
        if self.verbose:
            print(f"Iterations in AMG solver: {iteration_count[0]}.")

        return sol


class HessSolveSparse:
    def __init__(self, H):
        self.H = H

    def solve(self, x):
        x = spla.spsolve(self.H, x.copy())
        return x


class HessSolverGenerator:
    def __init__(self, ddf, solver_type: Literal["direct", "amg"] = "direct",
                 elastic_kernel=None, verbose=False, tol=1e-3, maxiter=100):
        self.ddf = ddf
        self.verbose = verbose
        self.tol = tol
        self.maxiter = maxiter
        self.elastic_kernel = elastic_kernel
        self.solver_type = solver_type

    def __call__(self, x):
        sparse_matrix_scipy = self.ddf(x)
        if self.solver_type == "direct":
            return HessSolveSparse(sparse_matrix_scipy)
        if self.solver_type == "amg":
            return SolverAMGElasticity(sparse_matrix_scipy, elastic_kernel=self.elastic_kernel, verbose=self.verbose,
                                       tol=self.tol, maxiter=self.maxiter)
        raise ValueError(f"Unknown type {self.solver_type} for the solver.")
