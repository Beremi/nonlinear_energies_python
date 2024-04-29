# Nonlinear Energies Python

This repository contains the accompanying code for the paper submitted to the **PPAM 2024** conference, titled "Minimization of Nonlinear Energies in Python Using FEM and Automatic Differentiation Tools" by Michal Béreš and Jan Valdman. The code, implemented in Python, leverages the JAX library to solve three numerical examples related to nonlinear energies.

**`The repository is set up for CodeSpaces use!`**

## Description of the Repository

- **tests**
  - `test_pLaplace2D.ipynb` - Tests for the p-Laplace problem in 2D.
  - `test_GinzburgLandau2D.ipynb` - Tests for the Ginzburg-Landau problem in 2D.
  - `test_HyperElasticity3D.ipynb` - Tests for the hyperelasticity problem in 3D.
- **benchmarks** - Corresponding to tables in the paper:
  - `benchmark_pLaplace2D.ipynb` - Benchmarks for the p-Laplace problem in 2D.
  - `benchmark_GinzburgLandau2D.ipynb` - Benchmarks for the Ginzburg-Landau problem in 2D.
  - `benchmark_HyperElasticity3D.ipynb` - Benchmarks for the hyperelasticity problem in 3D.

### Directories
- `tools` - Contains implementations of common tools, including graph coloring, differentiation using JAX, sparse solvers using PyAMG, and Newton's method.
- `pLaplace2D` - Contains the implementation of the p-Laplace problem in 2D, the energy operator in JAX, and mesh.
- `GinzburgLandau2D` - Contains the implementation of the Ginzburg-Landau problem in 2D, the energy operator in JAX, and mesh.
- `HyperElasticity3D` - Contains the implementation of the hyperelasticity problem in 3D, the energy operator in JAX, and mesh.
