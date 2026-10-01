import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from .attention import Attention
from .deprop_conv import DePropConv


class DePropEncoder(nn.Module):
    """
    Stack of DeProp layers with dropout, batch norm and activation.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    hid_dim : int
        Hidden feature dimensionality.
    out_dim : int
        Output feature dimensionality.
    num_layers : int
        Number of DeProp layers, at least ``2``.
    lambda1 : float
        Weight of the graph smoothing term.
    lambda2 : float
        Weight of the feature decorrelation term.
    gamma : float
        Step size of the propagation.
    dropout : float
        Dropout rate applied before every layer.
    act : callable
        Activation function.
    with_bn : bool
        Use batch norm after every hidden layer.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, lambda1, lambda2, gamma, dropout, act, with_bn):
        super().__init__()

        if num_layers < 2:
            raise ValueError('DePropEncoder needs at least 2 layers.')

        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([
            DePropConv(dims[i], dims[i + 1], lambda1, lambda2, gamma)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hid_dim) if with_bn else nn.Identity()
            for _ in range(num_layers - 1)
        ])
        self.dropout = dropout
        self.act = act

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim].
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges].
        edge_weight : torch.Tensor, optional
            Propagation weights [num_edges]. Default: ``None``.

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, out_dim].
        """
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(self.bns[i](x))
        return x


class GraphTransformerLayer(nn.Module):
    """
    Graph transformer layer with attention restricted to graph edges.

    Parameters
    ----------
    hid_dim : int
        Feature dimensionality, divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.

    Notes
    -----
    Post-norm block: edge-restricted multi-head attention with an output
    projection, then a two-layer feed-forward network, each with a residual
    connection followed by batch norm.
    """

    def __init__(self, hid_dim, num_heads):
        super().__init__()
        self.attention = TransformerConv(hid_dim, hid_dim // num_heads, heads=num_heads, root_weight=False)
        self.out_proj = nn.Linear(hid_dim, hid_dim)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.ffn1 = nn.Linear(hid_dim, hid_dim * 2)
        self.ffn2 = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, h, edge_index):
        h = self.bn1(h + self.out_proj(self.attention(h, edge_index)))
        return self.bn2(h + self.ffn2(F.relu(self.ffn1(h))))


class GraphTransformer(nn.Module):
    """
    Stack of graph transformer layers on top of Laplacian positional encodings.

    Parameters
    ----------
    hid_dim : int
        Feature dimensionality.
    pos_enc_dim : int
        Dimensionality of the positional encoding.
    num_layers : int
        Number of transformer layers.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, hid_dim, pos_enc_dim, num_layers, num_heads):
        super().__init__()
        self.pos_linear = nn.Linear(pos_enc_dim, hid_dim)
        self.layers = nn.ModuleList([GraphTransformerLayer(hid_dim, num_heads) for _ in range(num_layers)])

    def forward(self, h, edge_index, pos_enc):
        """
        Forward pass through the transformer.

        Parameters
        ----------
        h : torch.Tensor
            Node features [num_nodes, hid_dim].
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges].
        pos_enc : torch.Tensor
            Positional encoding [num_nodes, pos_enc_dim].

        Returns
        -------
        torch.Tensor
            Node features [num_nodes, hid_dim].
        """
        h = h + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(h, edge_index)
        return h


class DFTBase(nn.Module):
    """
    Base class for DFT.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of the DeProp encoder.
    emb_dim : int
        Output dimension of the encoder and width of the transformer.
    num_classes : int
        Number of classes.
    num_layers : int
        Number of DeProp layers.
    dropout : float
        Dropout rate of the DeProp encoder.
    act : callable activation function
        Activation function of the DeProp encoder.
    lambda1 : float
        Weight of the DeProp smoothing term.
    lambda2 : float
        Weight of the DeProp decorrelation term.
    gamma : float
        Step size of the DeProp propagation.
    with_bn : bool
        Use batch norm in the DeProp encoder.
    ppmi : bool
        Encode the PPMI view as well and fuse both views with attention.
    transformer : bool
        Refine the encoding with the graph transformer.
    pos_enc_dim : int
        Dimension of the Laplacian positional encoding.
    gt_layers : int
        Number of graph transformer layers.
    gt_heads : int
        Number of graph transformer attention heads.

    Notes
    -----
    Architecture Components:

    - DeProp encoder shared by the GCN and PPMI views
    - Attention fusion of the two views
    - Graph transformer
    - Classification head
    - Domain critic
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 emb_dim,
                 num_classes,
                 num_layers,
                 dropout,
                 act,
                 lambda1,
                 lambda2,
                 gamma,
                 with_bn,
                 ppmi,
                 transformer,
                 pos_enc_dim,
                 gt_layers,
                 gt_heads):
        super(DFTBase, self).__init__()

        self.ppmi = ppmi
        self.transformer = transformer

        self.encoder = DePropEncoder(
            in_dim, hid_dim, emb_dim, num_layers, lambda1, lambda2, gamma, dropout, act, with_bn)

        self.cls_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, num_classes)
        )

        self.domain_model = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        # The critic is trained by its own optimizer, so it is not in ``models``.
        self.models = [self.encoder, self.cls_model]

        if self.ppmi:
            self.att_model = Attention(emb_dim)
            self.models.append(self.att_model)

        if self.transformer:
            self.gt_model = GraphTransformer(emb_dim, pos_enc_dim, gt_layers, gt_heads)
            self.models.append(self.gt_model)

        self.loss_func = nn.CrossEntropyLoss()

    def encode(self, data):
        """
        Encode a graph prepared by ``DFT.process_graph``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        torch.Tensor
            Node embeddings [num_nodes, emb_dim].
        """
        encoded_output = self.encoder(data.x, data.gcn_edge_index, data.gcn_edge_weight)

        if self.ppmi:
            ppmi_output = self.encoder(data.x, data.ppmi_edge_index, data.ppmi_edge_weight)
            encoded_output = self.att_model([encoded_output, ppmi_output])

        if self.transformer:
            encoded_output = self.gt_model(encoded_output, data.edge_index, data.lap_pe)

        return encoded_output

    def domain_distance(self, encoded_source, encoded_target):
        """
        Gap between the mean critic scores of both domains.

        Parameters
        ----------
        encoded_source : torch.Tensor
            Source node embeddings.
        encoded_target : torch.Tensor
            Target node embeddings.

        Returns
        -------
        torch.Tensor
            Scalar critic gap.
        """
        return torch.abs(self.domain_model(encoded_source).mean() - self.domain_model(encoded_target).mean())

    def gradient_penalty(self, encoded_source, encoded_target):
        """
        Penalty pushing the critic's input-gradient norm towards one.

        Parameters
        ----------
        encoded_source : torch.Tensor
            Source node embeddings.
        encoded_target : torch.Tensor
            Target node embeddings.

        Returns
        -------
        torch.Tensor
            Scalar gradient penalty.

        Notes
        -----
        Evaluated at the embeddings themselves rather than at
        interpolations between domains, as in the reference implementation.
        """
        inputs = torch.vstack([encoded_source, encoded_target])
        if not inputs.requires_grad:
            inputs.requires_grad_(True)
        scores = self.domain_model(inputs)
        gradient = torch.autograd.grad(
            outputs=scores,
            inputs=inputs,
            grad_outputs=torch.ones_like(scores),
            create_graph=True
        )[0]
        return torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
