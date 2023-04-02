import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from IHGNN import IHGNN
from mlp_dropout import MLPClassifier
from sklearn import metrics
from util import cmd_args, load_data

class Classifier(nn.Module):
    def __init__(self, grid_config):
        super(Classifier, self).__init__()
        if cmd_args.model == 'IHGNN':
            model = IHGNN
        else:
            print('unknown model %s' % cmd_args.model)
            sys.exit()
        if cmd_args.model == 'IHGNN':
            self.gnn = model(num_layers=grid_config["num_layer"],
                            num_node_feats=cmd_args.feat_dim,
                            latent_dim=cmd_args.latent_dim,
                            k=cmd_args.sortpooling_k, 
                            conv1d_activation=cmd_args.conv1d_activation)
        out_dim = self.gnn.dense_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.final_hidden, num_class=cmd_args.num_class, with_dropout=cmd_args.dropout)


    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        node_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            node_feat.append(batch_graph[i].node_features)
            n_nodes += batch_graph[i].num_nodes
        node_feat = torch.cat(node_feat, 0)
        if cmd_args.device == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        node_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat)
        return self.mlp(embed, labels)


        

def experiment(g_list, classifier, sample_idxes, bsize, optimizer=None):
    total_loss = []
    total_iters = int((len(sample_idxes) - 1) / bsize) + 1
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    n_samples = 0

    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        logits, loss, acc = classifier(batch_graph)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = loss.data.cpu().detach().numpy()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
        total_loss.append( np.array([loss, acc]) * len(selected_idx))
        n_samples += len(selected_idx)

    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss, avg_acc = np.sum(total_loss, 0) / n_samples

    
    return avg_loss, avg_acc


if __name__ == '__main__':

    grid_config = {"num_layer":0,"batch_size":0,"learning_rate":0}
    num_layer_list = [3,5]
    batch_size_list = [32,64,128]
    learning_rate_list = [0.01, 0.001, 0.0001]

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    for num_layer in num_layer_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                grid_config["num_layer"] = num_layer
                grid_config["batch_size"] = batch_size
                grid_config["learning_rate"] = learning_rate

                for fold in range(1,11):
                    train_graphs, validation_graphs, test_graphs = load_data(cmd_args.degree_as_tag,fold)
                    print('# train: %d, # validation: %d, # test: %d' % (
                    len(train_graphs), len(validation_graphs), len(test_graphs)))
                    num_nodes_list = sorted([g.num_nodes for g in train_graphs + validation_graphs + test_graphs])
                    sortpooling_k = num_nodes_list[int(math.ceil( len(num_nodes_list))) - 1]
                    sortpooling_k = max(10, sortpooling_k)
                    cmd_args.sortpooling_k = sortpooling_k
                    print('k used in SortPooling is: ' + str(sortpooling_k))
                    classifier = Classifier(grid_config)
                    if cmd_args.device == 'gpu':
                        classifier = classifier.cuda()
                    optimizer = optim.Adam(classifier.parameters(), grid_config["learning_rate"])
                    acc_per_epoch = np.zeros((cmd_args.num_epochs, 2))
                    train_idxes = list(range(len(train_graphs)))

                    for epoch in range(cmd_args.num_epochs):
                        random.shuffle(train_idxes)
                        classifier.train()
                        train_loss, train_acc = experiment(train_graphs, classifier, train_idxes, bsize=grid_config["batch_size"], optimizer=optimizer)
                        print('Average training of epoch %d: loss %.5f acc %.5f \n' % (epoch, train_loss, train_acc))
                        classifier.eval()
                        val_loss, val_acc = experiment(validation_graphs, classifier, list(range(len(validation_graphs))),bsize=grid_config["batch_size"])
                        acc_per_epoch[epoch][0] = val_loss
                        print('Average validation of epoch %d: loss %.5f acc %.5f \n' % (epoch, val_loss, val_acc))
                        test_loss, test_acc = experiment(test_graphs, classifier, list(range(len(test_graphs))),bsize=grid_config["batch_size"])
                        acc_per_epoch[epoch][1] = test_acc
                        print('Average test of epoch %d: loss %.5f acc %.5f \n' % (epoch, test_loss, test_acc))

                    results_folder = 'results/result{}_{}_{}'.format(grid_config["num_layer"],grid_config["batch_size"],grid_config["learning_rate"])
                    if not os.path.exists(results_folder):
                        os.makedirs(results_folder)

                    with open(results_folder + '/' + cmd_args.data + '_acc_results.txt', 'a+') as f:
                        for epoch in range(cmd_args.num_epochs):
                            f.write(str(acc_per_epoch[epoch][0]) + ' ')
                            f.write(str(acc_per_epoch[epoch][1]) + '\n')

