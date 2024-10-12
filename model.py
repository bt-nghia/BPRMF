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