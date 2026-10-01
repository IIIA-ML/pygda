import sys
sys.path.append('..')

import random
import torch
import argparse
import time
import os.path as osp
import numpy as np

from pygda.datasets import CitationDataset, TwitchDataset
from pygda.datasets import AirportDataset, MAGDataset, BlogDataset

from pygda.models import DFT

from pygda.metrics import eval_micro_f1, eval_macro_f1, eval_roc_auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.transforms import OneHotDegree

parser = argparse.ArgumentParser()

# model agnostic params
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
parser.add_argument('--source', type=str, default='DE', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='EN', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

# model specific params
parser.add_argument('--emb_dim', type=int, default=128, help='encoder output and transformer dimension')
parser.add_argument('--lambda1', type=float, default=100., help='weight of the DeProp smoothing term')
parser.add_argument('--lambda2', type=float, default=0.001, help='weight of the DeProp decorrelation term')
parser.add_argument('--gamma', type=float, default=0.01, help='step size of the DeProp propagation')
parser.add_argument('--ppmi', action=argparse.BooleanOptionalAction, default=True, help='use the PPMI view or not')
parser.add_argument('--transformer', action=argparse.BooleanOptionalAction, default=True, help='use the graph transformer or not')
parser.add_argument('--pos_enc_dim', type=int, default=2, help='dimension of the Laplacian positional encoding')
parser.add_argument('--lambda_gp', type=float, default=10., help='weight of the critic gradient penalty')
parser.add_argument('--lambda_b', type=float, default=1., help='weight of the domain alignment loss')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# load data 
if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
elif args.source in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Twitch', args.source)
    source_dataset = TwitchDataset(path, args.source)
elif args.source in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', args.source)
    source_dataset = AirportDataset(path, args.source)
elif args.source in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Blog', args.source)
    source_dataset = BlogDataset(path, args.source)
elif args.source in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/MAG', args.source)
    source_dataset = MAGDataset(path, args.source)

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)
elif args.target in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target)
elif args.target in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', args.target)
    target_dataset = AirportDataset(path, args.target)
elif args.target in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Blog', args.target)
    target_dataset = BlogDataset(path, args.target)
elif args.target in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/MAG', args.target)
    target_dataset = MAGDataset(path, args.target)

# construct node attributes for Airport dataset
max_degree = 0
for name in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', name)
    dataset = AirportDataset(path, name)
    data_degree = max(degree(dataset[0].edge_index[0,:]))
    if data_degree > max_degree:
        max_degree = data_degree

if args.source in {'BRAZIL', 'USA', 'EUROPE'}:
    target_dataset.transform = OneHotDegree(int(max_degree))
    source_dataset.transform = OneHotDegree(int(max_degree))

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

if args.source not in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    if not is_undirected(source_data.edge_index):
        source_data.edge_index = to_undirected(source_data.edge_index)
    
    if not is_undirected(target_data.edge_index):
        target_data.edge_index = to_undirected(target_data.edge_index)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

# choose a graph domain adaptation model
model = DFT(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    emb_dim=args.emb_dim,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    lambda1=args.lambda1,
    lambda2=args.lambda2,
    gamma=args.gamma,
    ppmi=args.ppmi,
    transformer=args.transformer,
    pos_enc_dim=args.pos_enc_dim,
    lambda_gp=args.lambda_gp,
    lambda_b=args.lambda_b
    )

# train the model
model.fit(source_data, target_data)

# evaluate the performance
logits, labels = model.predict(target_data)

preds = logits.argmax(dim=1)

mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)

if args.source in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    auc = eval_roc_auc(labels, logits[:, 1])
else:
    auc = 0.0

results = 'dft,source,' + args.source + ',target,' + args.target + ',micro-f1,' + str(mi_f1) + ',macro-f1,' + str(ma_f1) + ',auc,' + str(auc)

with open(args.filename, 'a+') as f:
    f.write(results + '\n')

print(results)