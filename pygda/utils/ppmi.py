import numpy as np
import scipy.sparse as sp
import torch

from scipy.sparse import csc_matrix

__all__ = ["scale_sim_mat", "agg_tran_prob_mat", "compute_ppmi", "ppmi_edge_view"]


def scale_sim_mat(w):
    """
    Compute L1 row normalization of a matrix.

    Parameters
    ----------
    w : np.ndarray or scipy.sparse.csc_matrix
        Input matrix to be normalized.

    Returns
    -------
    np.ndarray or scipy.sparse.csc_matrix
        Row-normalized matrix.
    """
    rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum + 1e-12, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    w = r_mat_inv.dot(w)

    return w


def agg_tran_prob_mat(g, step):
    """
    Compute aggregated K-step transition probability matrix.

    Parameters
    ----------
    g : scipy.sparse.csc_matrix
        Graph adjacency matrix.
    step : int
        Number of transition steps.

    Returns
    -------
    np.ndarray
        Dense aggregated transition probability matrix.

    Notes
    -----
    Aggregates transition probabilities up to K steps for capturing
    higher-order proximity. The result is dense, so memory is quadratic
    in the number of nodes.
    """
    g = scale_sim_mat(g)
    g = csc_matrix.toarray(g)
    a_k = g
    a = g
    for k in np.arange(2, step+1):
        a_k = np.matmul(a_k, g)
        a = a+a_k/k

    return a


def compute_ppmi(a):
    """
    Compute Positive Pointwise Mutual Information (PPMI) matrix.

    Parameters
    ----------
    a : np.ndarray
        Aggregated transition probability matrix. Modified in place.

    Returns
    -------
    np.ndarray
        PPMI matrix.

    Notes
    -----
    PPMI captures the statistical significance of node co-occurrences
    in random walks.
    """
    np.fill_diagonal(a, 0)
    a = scale_sim_mat(a)
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p)*a) / (col[None, :]) + 1e-12)
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0

    return ppmi


def ppmi_edge_view(edge_index, num_nodes, step=3):
    """
    Build the row-normalized PPMI graph as a weighted edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices (2, num_edges).
    num_nodes : int
        Number of nodes in the graph.
    step : int, optional
        Number of transition steps aggregated before the PPMI.
        Default: ``3``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Contains:

        - PPMI edge indices (2, num_ppmi_edges)
        - Row-normalized PPMI edge weights (num_ppmi_edges,)

    Notes
    -----
    This is the ACDNE PPMI matrix, not the random-walk estimate used by
    ``PPMIConv``.
    """
    row, col = edge_index.detach().cpu().numpy()
    g = sp.csc_matrix(
        (np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
    # Duplicate edges would otherwise sum to weights above one.
    g.sum_duplicates()
    g.data[:] = 1.

    n_ppmi = sp.coo_matrix(scale_sim_mat(compute_ppmi(agg_tran_prob_mat(g, step))))
    ppmi_edge_index = torch.tensor(np.vstack([n_ppmi.row, n_ppmi.col]), dtype=torch.long)
    ppmi_edge_weight = torch.tensor(n_ppmi.data, dtype=torch.float32)

    return ppmi_edge_index.to(edge_index.device), ppmi_edge_weight.to(edge_index.device)
