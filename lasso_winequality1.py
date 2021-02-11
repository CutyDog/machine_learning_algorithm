import lasso
import numpy as np
import csv

# データ読み込み
Xy = []
with open("winequality-red.csv") as fp:
    for row in csv.reader(fp, delimiter=";"):
        Xy.append(row)
Xy = np.array(Xy[1:], dtype=np.float64)

# 訓練用データとテスト用データに分割する
np.random.seed(0)
np.random.shuffle(Xy)
train_X = Xy[:-1000, :-1]
train_y = Xy[:-1000, -1]
test_X = Xy[-1000:, :-1]
test_y = Xy[-1000:, -1]

# ハイパーパラメータを変えながら学習させて結果表示
for lambda_ in [1., 0.1, 0.01]:
    model = lasso.Lasso(lambda_)
    model.fit(train_X, train_y)
    y = model.predict(test_X)
    print("--- lambda = {} ---".format(lambda_))
    print("coefficients:")
    print(model.w_)
    mse = ((y - test_y)**2).mean()
    print("MSE: {:.3f}".format(mse))
