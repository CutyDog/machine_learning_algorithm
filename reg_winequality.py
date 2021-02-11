import linearreg
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

# 学習させる
model = linearreg.LinearRegression()
model.fit(train_X, train_y)

# テスト用データにモデルを適用
y = model.predict(test_X)

print("最初の5つの正解と予測値:")
for i in range(5):
    print("{:1.0f} {:5.3f}".format(test_y[i], y[i]))
print()
print("RMSE:", np.sqrt(((test_y - y)**2).mean()))
