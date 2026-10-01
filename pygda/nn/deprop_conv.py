import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class DePropConv(MessagePassing):
    """
    Decorrelated propagation layer from DeProp, as used by DFT.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    lambda1 : float
        Weight of the graph smoothing term.
    lambda2 : float
        Weight of the feature decorrelation term.
    gamma : float
        Step size of the propagation.
    bias : bool, optional
        Whether to use bias term. Default: ``True``.
    **kwargs : optional
        Additional arguments for MessagePassing base class.

    Notes
    -----
    With :math:`H = XW`, computes

    .. math::
        (1 - \\gamma\\lambda_1 + \\gamma\\lambda_2) H
        + \\gamma\\lambda_1 \\hat{A} H
        - \\gamma\\lambda_2 H H^\\top H + b,

    one gradient step on a smoothness plus decorrelation objective.
    ``edge_weight`` must already hold the propagation weights of
    :math:`\\hat{A}`; the layer does not normalize or add self-loops.
    """

    def __init__(self, in_channels, out_channels, lambda1, lambda2, gamma, bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        edge_weight : torch.Tensor, optional
            Propagation weights (num_edges,). Default: ``None`` (all ones).

        Returns
        -------
        torch.Tensor
            Output feature matrix (num_nodes, out_channels).
        """
        x = self.lin(x)
        out = (1 - self.gamma * self.lambda1 + self.gamma * self.lambda2) * x
        out = out + self.gamma * self.lambda1 * self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = out - self.gamma * self.lambda2 * x @ (x.t() @ x)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, lambda1={}, lambda2={}, gamma={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.lambda1, self.lambda2, self.gamma)
