import scipy.sparse as sp
from train import create_sp_graph
import pandas as pd
import numpy as np
import torch


nu = 10
ni = 8
k = 3 # k nearest user
topk = 4 # topk recommend item


ui_train_pairs = pd.read_csv(f"user_item_train.csv", sep="\t", names=None, header=None).to_numpy()
ui_test_pairs = pd.read_csv(f"user_item_test.csv", sep="\t", names=None, header=None).to_numpy()
ui_valid_pairs = pd.read_csv(f"user_item_valid.csv", sep="\t", names=None, header=None).to_numpy()


ui_train_graph = create_sp_graph(ui_train_pairs, shape=(nu, ni)).tocsr()
ui_test_graph = create_sp_graph(ui_test_pairs, (nu, ni)).tocsr()
ui_valid_graph = create_sp_graph(ui_valid_pairs, (nu, ni)).tocsr()
norm_weight = ui_train_graph.mean(axis=1) + 1e-8
# norm_ui_graph = ui_train_graph - ui_train_graph / norm_weight
norm_ui_graph = ui_train_graph - ui_train_graph / norm_weight
# print("ui_train", norm_ui_graph)


def cosine(mat):
    ovl = mat @ mat.T #[u x u]
    norm = np.sqrt((mat.multiply(mat)).sum(axis=1))
    norm = norm * norm.T
    cos_score = ovl / (norm + 1e-12)
    return cos_score


cosine_sim_u = cosine(ui_train_graph).tocsr()
cosine_sim_i = cosine(ui_train_graph.T).tocsr()


# evaluate as user KNN
recall_cnt = 0
pre_cnt = 0
non_pos_u = 0

for uid in range(0, nu):
    gdu = ui_test_graph[uid].nonzero()[1]
    if len(gdu) == 0:
        non_pos_u+=1
        continue
    # print(gdu)
    _, knn_uid_ids = torch.topk(torch.tensor(cosine_sim_u[uid].todense()), k=k+1)
    knn_uid_ids = knn_uid_ids.squeeze()[1:]
    knn_rating = ui_train_graph[knn_uid_ids]
    # print(knn_uid_ids)
    meanknn_rating = torch.tensor(knn_rating.mean(axis=0))
    # print(meanknn_rating)
    _, rec_item_ids = torch.topk(meanknn_rating, k=topk)
    # print(gdu, rec_item_ids)
    count = np.in1d(rec_item_ids, gdu).sum()
    # print(count)
    recall_cnt += count / len(gdu)
    pre_cnt += count / topk

recall_cnt /= (nu-non_pos_u)
pre_cnt /= (nu-non_pos_u)

print("UserKNN")
print("Recall@%i: %.4f, Precision@%i: %.4f" %(topk, recall_cnt, topk, pre_cnt))


# evaluate as ItemKNN
recall_cnt = 0
pre_cnt = 0
non_pos_u = 0

for uid in range(0, nu):
    gdu = ui_test_graph[uid].nonzero()[1]
    if len(gdu) == 0:
        non_pos_u+=1
        continue
    # print(gdu)
    _, knn_uid_ids = torch.topk(torch.tensor(cosine_sim_u[uid].todense()), k=k+1)
    knn_uid_ids = knn_uid_ids.squeeze()[1:]
    knn_rating = ui_train_graph[knn_uid_ids]
    # print(knn_uid_ids)
    meanknn_rating = torch.tensor(knn_rating.mean(axis=0))
    # print(meanknn_rating)
    _, rec_item_ids = torch.topk(meanknn_rating, k=topk)
    # print(gdu, rec_item_ids)
    count = np.in1d(rec_item_ids, gdu).sum()
    # print(count)
    recall_cnt += count / len(gdu)
    pre_cnt += count / topk

recall_cnt /= (nu-non_pos_u)
pre_cnt /= (nu-non_pos_u)

print("ItemKNN")
print("Recall@%i: %.4f, Precision@%i: %.4f" %(topk, recall_cnt, topk, pre_cnt))
