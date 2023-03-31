from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pdb
from mlp import MLP

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init, gnn_spmm


class IHGNN(nn.Module):
    def __init__(self,num_layers,  num_node_feats, latent_dim=32, k=30, conv1d_activation='ReLU'):
        print('Initializing IHGNN')
        super(IHGNN, self).__init__()
        self.latent_dim = latent_dim
        self.num_node_feats = num_node_feats
        self.k = k
        self.num_layers = num_layers
        self.alpha = nn.Parameter(torch.zeros(self.num_layers+1))
        self.mlps = torch.nn.ModuleList()
        self.mlps.append(MLP(2, num_node_feats, self.latent_dim, self.latent_dim))
        for i in range(0, self.num_layers):
            self.mlps.append(MLP(2, 3 * self.latent_dim, self.latent_dim, self.latent_dim))
        self.dense_dim = self.k  * self.latent_dim
        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, graph_list, node_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        n2n_sp, _, _ = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()

        node_feat = Variable(node_feat)
        n2n_sp = Variable(n2n_sp)
        h = self.aggregate_combine(node_feat, n2n_sp,  graph_sizes)

        return h

    def aggregate_combine(self, node_feat, n2n_sp,  graph_sizes):
        ''' graph convolution layers '''
        cur_message_layer = node_feat
        cat_message_layers = []
        ego = self.mlps[0](cur_message_layer)
        cat_message_layers.append(torch.mul(self.alpha[0],ego))

        for layer in range(1, self.num_layers + 1):
            neig = gnn_spmm(n2n_sp, ego) # Y = A * Y
            agg = torch.cat((ego, neig), 1)
            neig_ego = neig + ego
            agg = torch.cat((agg, neig_ego), 1)
            ego = self.mlps[layer](agg)
            cat_message_layers.append(torch.mul(self.alpha[layer], ego))

        out = torch.stack(cat_message_layers, 0)
        out = torch.sum(out,0)


        wl_color = ego[:, -1]
        batch_sort_graphs = torch.zeros(len(graph_sizes), self.k, self.latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sort_graphs = batch_sort_graphs.cuda()

        batch_sort_graphs = Variable(batch_sort_graphs)
        accum_count = 0
        for i in range(len(graph_sizes)):
            to_sort = wl_color[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sort_graph = out.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sort_graph = torch.cat((sort_graph, to_pad), 0)
            batch_sort_graphs[i] = sort_graph
            accum_count += graph_sizes[i]

        to_dense = batch_sort_graphs.view((-1, 1, self.k * self.latent_dim))
        to_dense = to_dense.view(len(graph_sizes), -1)
        reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
