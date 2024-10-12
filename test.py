import pandas as pd
import numpy as np

col1 = [0, 0, 1, 1, 3, 5, 5, 4, 4]
col2 = [0, 1, 2, 3, 2, 2, 3, 1, 3]

data = {
    "user": col1,
    "item": col2
}

df = pd.DataFrame(data=data)
print(df)
df.to_csv("user_item_train.csv", sep="\t", header=False, index=False)

g = pd.read_csv("user_item_train.csv", sep="\t", names=None)
g = g.to_numpy()
print(g)


utest = np.unique(g[:, 0])
print(utest)