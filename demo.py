import torch

topk = 10

uids = [1]
uids = torch.tensor(uids)

model = torch.load("weight.pth")
score = model.pred(uids)
ranking_score, rec_ids = torch.topk(score, k=topk, dim=1)
print(rec_ids)