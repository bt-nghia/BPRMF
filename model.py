import torch
from torch import nn


class BPRMF(nn.Module):
    def __init__(self, nu, ni, nd):
        super().__init__()
        self.nu = nu
        self.ni = ni
        self.nd = nd
        self.init_emb()

    def init_emb(self):
        self.user_emb = nn.Parameter(torch.FloatTensor(self.nu, self.nd))
        nn.init.xavier_normal_(self.user_emb)
        self.item_emb = nn.Parameter(torch.FloatTensor(self.ni, self.nd))
        nn.init.xavier_normal_(self.item_emb)

    @torch.no_grad
    def pred(self, uids):
        score = self.user_emb[uids] @ self.item_emb.T
        return score

    def forward(self, X):
        '''
        BPR loss
        '''
        uids, piids, niids = X[:, 0], X[:, 1], X[:, 2]
        pos_score = torch.sum(self.user_emb[uids] * self.item_emb[piids], axis=1)
        neg_score = torch.sum(self.user_emb[uids] * self.item_emb[niids], axis=1)
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
    
    def pred(self, users):
        # print(users)
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