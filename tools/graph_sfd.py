import igraph
import scipy.sparse as sps
import numpy as np


def color_connectivity_for_adjacency(adjacency: sps.coo_matrix) -> tuple[np.ndarray,
                                                                         tuple[np.ndarray, np.ndarray],
                                                                         tuple[np.ndarray, np.ndarray]]:
    """
    Calculate the coloring of connectivity based on an adjacency matrix.

    Parameters:
    adjacency_matrix (sps.coo_matrix): The adjacency matrix of a graph in COO format.

    Returns:
    tuple: (vertex coloring array,
           the compressed index for block of Hessian tangent dots,
           and the sparse index for final sparse Hessian construction.
    """
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    n = adjacency.shape[0]
    connectivity = sps.coo_array(adjacency @ adjacency)
    graph_connectivity = igraph.Graph()
    graph_connectivity.add_vertices(n)
    indices = np.array(connectivity.coords).T
    graph_connectivity.add_edges(indices)
    coloring = graph_connectivity.vertex_coloring_greedy()
    coloring = np.array(coloring, dtype=np.int64).ravel()
    row, col = adjacency.nonzero()
    compressed_index = (row, coloring[col])
    sparse_index = (row, col)
    return coloring, compressed_index, sparse_index


def coloring_to_grouping(coloring: np.ndarray) -> np.ndarray:
    """
    Convert a vertex coloring array to a group matrix where each row represents a vertex
    and each column a group. The entries are 1 if the vertex belongs to the group, otherwise 0.

    Parameters:
    vertex_coloring (np.ndarray): Array of vertex colors.

    Returns:
    np.ndarray: float matrix indicating group memberships.
    """
    groups = np.max(coloring) + 1
    group_vec = np.arange(groups)[None, :]
    grouping: np.ndarray = (group_vec == coloring[:, None])
    return grouping.astype(np.float64)
