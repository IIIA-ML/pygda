import copy
import itertools
import time

import torch
import torch.nn.functional as F

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from . import BaseGDA
from ..nn import DFTBase
from ..utils import logger, ppmi_edge_view
from ..metrics import eval_micro_f1


class DFT(BaseGDA):
    """
    Enhancing Node-Level Graph Domain Adaptation by Alleviating Local Dependency (KDD-26).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of the DeProp encoder.
    num_classes : int
        Total number of classes.
    emb_dim : int, optional
        Output dimension of the encoder and width of the graph transformer.
        Default: ``128``.
    num_layers : int, optional
        Total number of DeProp layers. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.5``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lambda1 : float, optional
        Weight of the DeProp smoothing term. Default: ``100.``.
    lambda2 : float, optional
        Weight of the DeProp decorrelation term. Default: ``0.001``.
    gamma : float, optional
        Step size of the DeProp propagation. Default: ``0.01``.
    improved : bool, optional
        Use :math:`A + 2I` instead of :math:`A + I` in the GCN view.
        Default: ``True``.
    with_bn : bool, optional
        Use batch norm in the DeProp encoder. Default: ``True``.
    ppmi : bool, optional
        Add the PPMI view and fuse it with attention. Default: ``True``.
    ppmi_step : int, optional
        Number of transition steps aggregated for the PPMI. Default: ``3``.
    transformer : bool, optional
        Use the graph transformer. Default: ``True``.
    pos_enc_dim : int, optional
        Dimension of the Laplacian positional encoding. Default: ``2``.
    gt_layers : int, optional
        Number of graph transformer layers. Default: ``4``.
    gt_heads : int, optional
        Number of graph transformer attention heads. Default: ``1``.
    critic_iters : int, optional
        Critic updates per training epoch. Default: ``10``.
    lambda_gp : float, optional
        Weight of the critic gradient penalty. Default: ``10.``.
    lambda_b : float, optional
        Weight of the domain alignment loss. Default: ``1.``.
    weight_mean_reg : float, optional
        Weight of the mean-of-weights term added to the classification loss.
        Default: ``0.003``.
    entropy_weight : float, optional
        Final weight of the target entropy loss, reached linearly.
        Default: ``0.01``.
    lr : float, optional
        Learning rate. Default: ``0.003``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``500``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Only full batch training (``0``) is supported. Default: ``0``.
    num_neigh : int, optional
        Unused, as training is full batch. Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.

    Notes
    -----
    Departures from the reference implementation:

    - The GCN view of each graph is normalized on that graph. The reference
      caches the normalization of the first graph it sees and reuses it for
      the target graph.
    - Training keeps the final model, without selecting the epoch by
      target accuracy.

    ``improved=True`` and ``weight_mean_reg`` reproduce the reference code.
    The former is set there by passing ``orth`` positionally into the
    ``improved`` slot; the latter adds the mean, not a norm, of every
    weight tensor to the loss.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        emb_dim=128,
        num_layers=3,
        dropout=0.5,
        weight_decay=0.,
        act=F.relu,
        lambda1=100.,
        lambda2=1e-3,
        gamma=0.01,
        improved=True,
        with_bn=True,
        ppmi=True,
        ppmi_step=3,
        transformer=True,
        pos_enc_dim=2,
        gt_layers=4,
        gt_heads=1,
        critic_iters=10,
        lambda_gp=10.,
        lambda_b=1.,
        weight_mean_reg=3e-3,
        entropy_weight=0.01,
        lr=3e-3,
        epoch=500,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):

        super(DFT, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            weight_decay=weight_decay,
            act=act,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs)

        if batch_size != 0:
            raise ValueError('DFT only supports full batch training (batch_size=0).')

        self.emb_dim = emb_dim
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma = gamma
        self.improved = improved
        self.with_bn = with_bn
        self.ppmi = ppmi
        self.ppmi_step = ppmi_step
        self.transformer = transformer
        self.pos_enc_dim = pos_enc_dim
        self.gt_layers = gt_layers
        self.gt_heads = gt_heads
        self.critic_iters = critic_iters
        self.lambda_gp = lambda_gp
        self.lambda_b = lambda_b
        self.weight_mean_reg = weight_mean_reg
        self.entropy_weight = entropy_weight

    def init_model(self, **kwargs):
        """
        Initialize the DFT model.

        Parameters
        ----------
        **kwargs
            Other parameters for the DFTBase model.

        Returns
        -------
        DFTBase
            Initialized DFT model on the specified device.
        """

        return DFTBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            emb_dim=self.emb_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            gamma=self.gamma,
            with_bn=self.with_bn,
            ppmi=self.ppmi,
            transformer=self.transformer,
            pos_enc_dim=self.pos_enc_dim,
            gt_layers=self.gt_layers,
            gt_heads=self.gt_heads,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, epoch, optimizer_critic):
        """
        Forward pass of the model, including the critic updates.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Processed source domain graph data.
        target_data : torch_geometric.data.Data
            Processed target domain graph data.
        epoch : int
            Current training epoch.
        optimizer_critic : torch.optim.Optimizer
            Optimizer of the domain critic.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Combined loss from classification, domain alignment, and entropy.
            - source_logits : torch.Tensor
                Model predictions for source domain.

        Notes
        -----
        The critic is first trained for ``critic_iters`` steps on detached
        embeddings to maximize the gap between its mean scores on both
        domains, with a gradient penalty. The returned loss combines

        - Source classification loss
        - Critic gap under the updated critic
        - Target entropy minimization loss
        """
        encoded_source = self.dft.encode(source_data)
        encoded_target = self.dft.encode(target_data)

        detached_source = encoded_source.detach()
        detached_target = encoded_target.detach()
        for _ in range(self.critic_iters):
            loss_critic = (
                -self.dft.domain_distance(detached_source, detached_target)
                + self.lambda_gp * self.dft.gradient_penalty(detached_source, detached_target)
            )
            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

        loss_domain = self.dft.domain_distance(encoded_source, encoded_target)

        source_logits = self.dft.cls_model(encoded_source)
        cls_loss = self.dft.loss_func(source_logits, source_data.y)

        weight_mean = sum(
            param.mean()
            for model in self.dft.models
            for name, param in model.named_parameters()
            if 'weight' in name
        )
        cls_loss = cls_loss + self.weight_mean_reg * weight_mean

        loss = cls_loss + self.lambda_b * loss_domain

        target_logits = self.dft.cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy * (epoch / self.epoch * self.entropy_weight)

        return loss, source_logits

    def fit(self, source_data, target_data):
        """
        Train the DFT model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process includes:

        - Building the GCN, PPMI and positional encoding inputs of both graphs
        - Full batch training on both domains
        - Alternating critic and model updates
        - Gradually increasing the target entropy weight
        """
        self.processed_data = {
            'source': self.process_graph(source_data),
            'target': self.process_graph(target_data),
        }
        source_data = self.processed_data['source']
        target_data = self.processed_data['target']

        self.dft = self.init_model(**self.kwargs)

        params = itertools.chain(*[model.parameters() for model in self.dft.models])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        optimizer_critic = torch.optim.Adam(self.dft.domain_model.parameters(), lr=self.lr)

        start_time = time.time()

        for epoch in range(self.epoch):
            self.dft.train()

            loss, source_logits = self.forward_model(source_data, target_data, epoch, optimizer_critic)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            micro_f1_score = eval_micro_f1(source_data.y, source_logits.argmax(dim=1))

            logger(epoch=epoch,
                   loss=loss.item(),
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        torch_geometric.data.Data
            Shallow copy on ``self.device`` with the extra inputs of DFT:

            - ``gcn_edge_index``, ``gcn_edge_weight``: normalized GCN view
            - ``ppmi_edge_index``, ``ppmi_edge_weight``: row-normalized PPMI view
            - ``lap_pe``: Laplacian eigenvector positional encoding
        """
        data = copy.copy(data).to(self.device)

        data.gcn_edge_index, data.gcn_edge_weight = gcn_norm(
            data.edge_index, None, data.num_nodes, improved=self.improved, add_self_loops=True)

        if self.ppmi:
            data.ppmi_edge_index, data.ppmi_edge_weight = ppmi_edge_view(
                data.edge_index, data.num_nodes, self.ppmi_step)

        if self.transformer:
            data = AddLaplacianEigenvectorPE(self.pos_enc_dim, attr_name='lap_pe', is_undirected=True)(data)
            data.lap_pe = data.lap_pe.to(self.device)

        return data

    def predict(self, data, source=False):
        """
        Make predictions on the graphs given to ``fit``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data, the source or target graph given to ``fit``.
        source : bool, optional
            Whether the input is from source domain.
            Default: ``False``.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions.
            - labels : torch.Tensor
                True labels.
        """
        processed = self.processed_data['source' if source else 'target']
        if processed.num_nodes != data.num_nodes:
            raise ValueError('DFT.predict only supports the graphs given to fit.')

        self.dft.eval()

        with torch.no_grad():
            logits = self.dft.cls_model(self.dft.encode(processed))

        return logits, processed.y
