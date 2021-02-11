import matplotlib.pyplot as plt
import csv
import pca


# データ読み込み
Xy = []
with open("winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)
Xy = np.array(Xy[1:], dtype=np.float64)
X = Xy[:, :-1]

# 学習
model = pca.PCA(n_components=2)
model.fit(X)

# 変換
Y = model.transform(X)

# 描画
plt.scatter(Y[:, 0], Y[:, 1], color="k")
plt.show()
