import scipy.sparse as sp
from train import create_sp_graph
import pandas as pd
import numpy as np


nu = 10
ni = 8


ui_train_pairs = pd.read_csv(f"user_item_train.csv", sep="\t", names=None).to_numpy()
ui_test_pairs = pd.read_csv(f"user_item_test.csv", sep="\t", names=None).to_numpy()
ui_valid_pairs = pd.read_csv(f"user_item_valid.csv", sep="\t", names=None).to_numpy()


ui_train_graph = create_sp_graph(ui_train_pairs, shape=(nu, ni)).tocsr()
ui_test_graph = create_sp_graph(ui_test_pairs, (nu, ni)).tocsr()
ui_valid_graph = create_sp_graph(ui_valid_pairs, (nu, ni)).tocsr()

norm_weight = ui_train_graph.mean(axis=1) + 1e-8
# norm_ui_graph = ui_train_graph - ui_train_graph / norm_weight
norm_ui_graph = ui_train_graph - ui_train_graph / norm_weight
# print("ui_train", norm_ui_graph)


# print("ui_train_graph")
# print(ui_train_graph.todense())

def cosine(mat):
    # print(mat.shape)
    ovl = mat @ mat.T #[u x u]
    # print("ovl")
    # print(ovl.todense())
    # print(mat.todense())
    # print((mat.multiply(mat)).todense())
    norm = np.sqrt((mat.multiply(mat)).sum(axis=1))
    # print(norm)
    norm = norm * norm.T
    # print(norm)
    cos_score = ovl / (norm + 1e-12)
    # print(cos_score)
    return cos_score


print(cosine(ui_train_graph))