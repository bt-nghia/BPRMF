import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv, SAGEConv
from torch_geometric.nn.models import GraphSAGE, PMLP, GAT, LINKX, EdgeCNN


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


class BPRMF(nn.Module):
    def __init__(self, nu, ni, nd, ui_graph):
        super().__init__()
        self.nu = nu
        self.ni = ni
        self.nd = nd
        self.ui_graph = ui_graph
        self.init_emb()

        self.device = "cuda"

        self.n_layers = 2
        # MLP archi
        self.u_layer = nn.Sequential(
            nn.Linear(nd, nd*2, bias=False),
            nn.ReLU(),
            nn.Linear(nd * 2, nd, bias=False),
        )

        self.i_layer = nn.Sequential(
            nn.Linear(nd, nd*2, bias=False),
            nn.ReLU(),
            nn.Linear(nd * 2, nd, bias=False),
        )

        # self.create_graph()
        # print(ui_graph.tocoo().row, ui_graph.tocoo().col)
        # exit()
        ui_dense = ui_graph.tocoo()
        user_index = ui_dense.row
        item_index = ui_dense.col + self.nu

        # print(item_index.max())
        # print(user_index)
        # print(item_index)
        # print(user_index.shape)
        # print(item_index.shape)
        # exit()

        src_ids = np.concatenate((user_index, item_index), axis=0)
        trg_ids = np.concatenate((item_index, user_index), axis=0)

        self.edge_index = torch.LongTensor([src_ids, trg_ids]).to(self.device)
        # print(self.edge_index.shape)
        # print(user_index.max())
        # print(item_index.max())
        # print(len(user_index))
        # print(len(item_index))
        # print(user_index)
        # print(item_index)
        # print(user_index + item_index)
        # exit()
        # self.GAT_model = GAT(in_channels=self.nd, hidden_channels=self.nd * 2, num_layers=self.n_layers, out_channels=self.nd, dropout=0.1)
        # self.PMLP_model = PMLP(in_channels=self.nd, hidden_channels=self.nd * 2, num_layers=self.n_layers, out_channels=self.nd, dropout=0.1)
        # self.GraphSage_model = GraphSAGE(in_channels=self.nd, hidden_channels=self.nd * 2, num_layers=self.n_layers, out_channels=self.nd, dropout=0.1)
        self.LinkX_model = LINKX(self.nu + self.ni, self.nd, self.nd, self.nd, self.n_layers)
        self.edgecnn_model = EdgeCNN(in_channels=self.nd, hidden_channels=self.nd * 2, num_layers=self.n_layers, out_channels=self.nd, dropout=0.1)

    def create_graph(self):
        ui_propagate_graph = sp.bmat([[sp.csr_matrix((self.ui_graph.shape[0], self.ui_graph.shape[0])), self.ui_graph], 
                                      [self.ui_graph.T, sp.csr_matrix((self.ui_graph.shape[1], self.ui_graph.shape[1]))]])
        self.ui_propagate_graph = to_tensor(laplace_transform(ui_propagate_graph)).to(self.device)


    def init_emb(self):
        self.user_emb = nn.Parameter(torch.FloatTensor(self.nu, self.nd))
        nn.init.xavier_normal_(self.user_emb)
        self.item_emb = nn.Parameter(torch.FloatTensor(self.ni, self.nd))
        nn.init.xavier_normal_(self.item_emb)

    def propagate(self):
        # u_feat = self.u_layer(self.user_emb)
        # i_feat = self.i_layer(self.item_emb)

        # com_feat = torch.cat([u_feat, i_feat], dim=0)
        # feats = [com_feat]
        # for i in range(self.n_layers):
        #     com_feat = self.ui_propagate_graph @ com_feat / (i+2)
        #     feats.append(F.normalize(com_feat, p=2, dim=1))
        
        # feats = torch.stack(feats, dim=1)
        # feats = torch.sum(feats, dim=1).squeeze(1)
        # u_feat, i_feat = torch.split(feats, [self.nu, self.ni], dim=0)

        feats = torch.cat([self.user_emb, self.item_emb], dim=0)
        # out_feats = self.GAT_model(feats, self.edge_index)
        # out_feats = self.PMLP_model(feats, self.edge_index)
        # out_feats = self.GraphSage_model(feats, self.edge_index)
        # out_feats = self.LinkX_model(feats, self.edge_index)
        out_feats = self.edgecnn_model(feats, self.edge_index)

        u_feat, i_feat = torch.split(out_feats, self.user_emb.shape[0])
        return u_feat, i_feat

    @torch.no_grad
    def pred(self, uids):
        u_feat, i_feat = self.propagate()
        # score = self.user_emb[uids] @ self.item_emb.T
        score = u_feat[uids] @ i_feat.T
        return score

    def forward(self, X):
        '''
        BPR loss
        '''
        u_feat, i_feat = self.propagate()
        uids, piids, niids = X[:, 0], X[:, 1], X[:, 2]
        pos_score = torch.sum(u_feat[uids] * i_feat[piids], axis=1)
        neg_score = torch.sum(u_feat[uids] * i_feat[niids], axis=1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score))
        loss = torch.mean(loss)
        return loss
    
    def loss_func(self, X):
        return self(X)

# PreferedAI

import torch
import torch.nn as nn


optimizer_dict = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

activation_functions = {
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "leakyrelu": nn.LeakyReLU(),
}


class GMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
    ):
        super(GMF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, num_factors)
        self.item_embedding = nn.Embedding(num_items, num_factors)

        self.logit = nn.Linear(num_factors, 1)
        self.Sigmoid = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=1e-2)
        nn.init.normal_(self.item_embedding.weight, std=1e-2)
        nn.init.normal_(self.logit.weight, std=1e-2)

    def h(self, users, items):
        return self.user_embedding(users) * self.item_embedding(items)

    def forward(self, users, items):
        h = self.h(users, items)
        output = self.Sigmoid(self.logit(h)).view(-1)
        return output


class MLP(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        layers=(64, 32, 16, 8),
        act_fn="relu",
    ):
        super(MLP, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)

        mlp_layers = []
        for idx, factor in enumerate(layers[:-1]):
            mlp_layers.append(nn.Linear(factor, layers[idx + 1]))
            mlp_layers.append(activation_functions[act_fn.lower()])

        # unpacking layers in to torch.nn.Sequential
        self.mlp_model = nn.Sequential(*mlp_layers)

        self.logit = nn.Linear(layers[-1], 1)
        self.Sigmoid = nn.Sigmoid()

        self._init_weight()

    def _init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std=1e-2)
        nn.init.normal_(self.item_embedding.weight, std=1e-2)
        for layer in self.mlp_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.logit.weight, std=1e-2)

    def h(self, users, items):
        embed_user = self.user_embedding(users)
        embed_item = self.item_embedding(items)
        embed_input = torch.cat((embed_user, embed_item), dim=-1)
        return self.mlp_model(embed_input)

    def forward(self, users, items):
        h = self.h(users, items)
        output = self.Sigmoid(self.logit(h)).view(-1)
        return output

    def __call__(self, *args):
        return self.forward(*args)


class NeuMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
    ):
        super(NeuMF, self).__init__()
        self.nu = num_users
        self.ni = num_items
        # layer for MLP
        if layers is None:
            layers = [64, 32, 16, 8]
        if num_factors is None:
            num_factors = layers[-1]

        assert layers[-1] == num_factors

        self.logit = nn.Linear(num_factors + layers[-1], 1)
        self.Sigmoid = nn.Sigmoid()

        self.gmf = GMF(num_users, num_items, num_factors)
        self.mlp = MLP(
            num_users=num_users, num_items=num_items, layers=layers, act_fn=act_fn
        )

        nn.init.normal_(self.logit.weight, std=1e-2)

    def forward(self, users, items, gmf_users=None):
        # gmf_users is there to take advantage of broadcasting
        h_gmf = (
            self.gmf.h(users, items)
            if gmf_users is None
            else self.gmf.h(gmf_users, items)
        )
        h_mlp = self.mlp.h(users, items)
        h = torch.cat([h_gmf, h_mlp], dim=-1)
        output = self.Sigmoid(self.logit(h)).view(-1)
        return output
    
    def loss_func(self, X, gmf_users=None):
        users, pitems, nitems = X[:,0], X[:,1], X[:,2]
        pos_score = self(users, pitems)
        neg_score = self(users, nitems)
        loss = - torch.log(pos_score) - torch.log(1-neg_score)
        loss = loss.mean()
        return loss
    
    @torch.no_grad
    def pred(self, users):
        score = []
        all_iids = torch.arange(0, self.ni)
        for uid in users:
            u_score = []
            for iid in all_iids:
                uid_score = self(torch.tensor([uid]), torch.tensor([iid]))
                u_score.append(uid_score)
            score.append(u_score)
        score = torch.tensor(score)
        return score
    

class Pop(nn.Module):
    def __init__(self, ui_graph, nu, ni):
        super().__init__()
        self.ui_graph = ui_graph
        self.pop_item = ui_graph.T.sum(axis = 0) # return item popularity for everyone

    # please remove training part if using this model
    def pred(self, users):
        # user: tensor([0, 1, 4, ....])
        return torch.tensor(self.pop_item).expand(users.shape[1], -1)
