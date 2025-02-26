import torch
import numpy as np
import pandas as pd
from torch import optim
from model import BPRMF, NeuMF
from scipy import sparse as sp
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

INF = 1e12
EPSILON = 1e-8

conf = {
    "nu": 6041,
    "ni": 3953,
    "nd": 16,
    "epoch": 100,
    "lr": 1e-3,
    "n_log": 5,
    "batch_size": 4096,
    "topk": [20, 30, 50],
}

def read_csv(path):
    data = pd.read_csv(path, sep=",")
    interaction = data[["user_id", "item_id"]]
    interaction = interaction.to_numpy()
    return interaction


def create_sp_graph(index_pairs, shape):
    assert index_pairs.shape[1] == 2
    values = np.ones(index_pairs.shape[0], dtype=np.int32)
    sp_graph = sp.coo_matrix(
        (values, (index_pairs[:,0], index_pairs[:,1])),
        shape=shape
    )
    norm_sp_graph = (sp_graph > 0)*1
    return norm_sp_graph


class UserItemData(Dataset):
    def __init__(self, nu, ni):
        self.nu = nu
        self.ni = ni
        # self.ui_pairs = pd.read_csv(f"ml-1m/user_item_train.csv", sep="\t", names=None, header=None).to_numpy()
        self.ui_pairs = read_csv("ml-1m/train 1.csv")
        self.ui_graph = create_sp_graph(self.ui_pairs, (nu, ni))

    def __getitem__(self, index):
        uid, piid = self.ui_pairs[index]
        while 1:
            niid = np.random.randint(0, self.ni)
            if self.ui_graph[uid, niid] == 0:
                break
        return torch.LongTensor([uid, piid, niid])
    
    def __len__(self):
        return len(self.ui_pairs)
    

class UserItemTestData(Dataset):
    def __init__(self, nu, ni):
        self.nu = nu
        self.ni = ni
        # self.ui_test_pairs = pd.read_csv(f"ml-1m/user_item_test.csv", sep="\t", names=None, header=None).to_numpy()
        self.ui_test_pairs = read_csv("ml-1m/test 1.csv")
        self.ui_graph_test = create_sp_graph(self.ui_test_pairs, (nu, ni))

        # self.ui_valid_pairs = pd.read_csv(f"ml-1m/user_item_valid.csv", sep="\t", names=None, header=None).to_numpy()
        self.ui_valid_pairs = read_csv("ml-1m/valid 1.csv")
        self.ui_graph_valid = create_sp_graph(self.ui_valid_pairs, (nu, ni))

    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return self.nu


if __name__ == "__main__":
    torch.manual_seed(2024)
    np.random.seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ui_train_data = UserItemData(nu=conf["nu"], ni=conf["ni"])
    ui_valid_test_data = UserItemTestData(nu=conf["nu"], ni=conf["ni"])
    train_loader = DataLoader(ui_train_data, batch_size=conf["batch_size"], drop_last=False, shuffle=True)
    valid_test_loader = DataLoader(ui_valid_test_data, batch_size=conf["batch_size"], drop_last=False, shuffle=False)

    # ground truth for valid & test
    ui_train_graph = ui_train_data.ui_graph
    ui_valid_graph = ui_valid_test_data.ui_graph_valid.tocsr()
    ui_test_graph = ui_valid_test_data.ui_graph_test.tocsr()

    model = BPRMF(
        nu=conf["nu"],
        ni=conf["ni"],
        nd=conf["nd"],
        ui_graph=ui_train_graph).to(device)

    # model = NeuMF(
    #     num_users=conf["nu"],
    #     num_items=conf["ni"],
    # )

    torch.save(model, "weight.pth")

    
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=conf["lr"])
    
    for epoch in range(0, conf["epoch"]):
        model.train()
        pbar = tqdm(train_loader)
        for iter, data in enumerate(pbar):
            # print(data)
            data = data.to(device)
            optimizer.zero_grad()
            loss = model.loss_func(data)
            loss.backward()
            optimizer.step()

            pbar.set_description("epoch: %i, loss: %.4f" %(epoch, loss.detach()))


        if 1 == 1:
            # init metrics
            vrecall, trecall, vpre, tpre, vrscore, trscore, vprescore, tprescore = \
                {}, {}, {}, {}, {}, {}, {}, {}
            for topk in conf["topk"]:
                vrecall[topk] = []
                trecall[topk] = []
                vpre[topk] = []
                tpre[topk] = []

                vrscore[topk] = 0
                tprescore[topk] = 0
                trscore[topk] = 0
                vprescore[topk] = 0

            # evaluation
            model.eval()
            # recall, ndcg log
            recall_satck = []
            pre_stack = []
            for uids in valid_test_loader:
                # ranking score
                uids_score = model.pred(uids).to("cpu")
                # print(uids_score.shape)

                # rec_score = u_feat[uids] @ i_feat.T 
                # con_score = con_rec_mat[uids].todense()
                # uids_score = torch.tensor(rec_score) + torch.tensor(con_score)

                # uids_score += con_score

                # evaluate
                for topk in conf["topk"]:
                    mask_score = ui_train_graph[uids] * (-INF)
                    mask_score = mask_score.todense()
                    # mask_score.to("cpu")
                    _, col_ids = torch.topk(uids_score, k=topk, axis=1)
                    row_ids = torch.tensor(uids).expand(topk, uids.shape[0]).T.reshape(-1)
                    col_ids = col_ids.reshape(-1)
                    
                    valid_is_hit = ui_valid_graph[row_ids.tolist(), col_ids.tolist()].reshape(-1 , topk)
                    test_is_hit = ui_test_graph[row_ids.tolist(), col_ids.tolist()].reshape(-1, topk)

                    #recall_minibatch
                    vrecall_batch = valid_is_hit.sum(axis=1) / (ui_valid_graph[uids].sum(axis=1) + EPSILON)
                    trecall_batch = test_is_hit.sum(axis=1) / (ui_test_graph[uids].sum(axis=1) + EPSILON)
                    vrecall[topk].append(torch.tensor(vrecall_batch))
                    trecall[topk].append(torch.tensor(trecall_batch))
                    #ndcg_minibatch
                    vpre_batch = valid_is_hit.sum(axis=1) / topk
                    tpre_batch = test_is_hit.sum(axis=1) / topk
                    vpre[topk].append(torch.tensor(vpre_batch))
                    tpre[topk].append(torch.tensor(tpre_batch))

            # combine all batch score
            for topk in conf["topk"]:
                vrscore[topk] = torch.cat(vrecall[topk], dim=0).sum() / \
                    (conf["nu"] - (ui_valid_graph.sum(axis=1) == 0).sum())
                trscore[topk] = torch.cat(trecall[topk], dim=0).sum() / \
                    (conf["nu"] - (ui_test_graph.sum(axis=1) == 0).sum())
                
                vprescore[topk] = torch.cat(vpre[topk], dim=0).sum() / \
                    (conf["nu"] - (ui_valid_graph.sum(axis=1) == 0).sum())
                tprescore[topk] = torch.cat(tpre[topk], dim=0).sum() / \
                    (conf["nu"] - (ui_test_graph.sum(axis=1) == 0).sum())
                print("topk: %i, recall_V: %.4f, recall_T: %.4f" %(topk, vrscore[topk], trscore[topk]))
                print("topk: %i, pre_V: %.4f, pre_T: %.4f" %(topk, vprescore[topk], tprescore[topk]))
    
    torch.save(model, "weight.pth")
        # exit()


    # score = model.pred(uids=torch.LongTensor([0, 1, 2, 3, 4, 5]))
    # x, y = torch.topk(score, axis=1, k=2)
    # # print(y)
