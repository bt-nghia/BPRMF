import torch
import pandas as pd
import numpy as np

topk = 10

uids = [1] # single user only
uids = torch.tensor(uids)

model = torch.load("weight.pth")
score = model.pred(uids)
ranking_score, rec_ids = torch.topk(score, k=topk, dim=1)
print(rec_ids)

movies = pd.read_csv("ml-1m/movies.dat", sep="::", encoding = "latin-1", names=["id", "name", "cates"])
# print(movies[movies["id"] == 1])
movies = movies[["id", "name"]]

for i in range(len(uids)):
    print("Rec film for user", uids[i])
    for iid in rec_ids[i]:
        print(movies[movies["id"] == int(iid)])