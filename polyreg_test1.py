import polyreg
import linearreg
import numpy as np
import matplotlib.pyplot as plt

# データ生成
np.random.seed(0)


def f(x):
    return 1 + 2 * x


x = np.random.random(10) * 10
y = f(x) + np.random.randn(10)

# 多項式回帰
model = polyreg.PolynomialRegression(10)
model.fit(x, y)

plt.scatter(x, y, color="k")
plt.ylim([y.min() - 1, y.max() + 1])
xx = np.linspace(x.min(), x.max(), 300)
yy = np.array([model.predict(u) for u in xx])
plt.plot(xx, yy, color="k")

# 線形回帰
model = linearreg.LinearRegression()
model.fit(x, y)
b, a = model.w_
x1 = x.min() - 1
x2 = x.max() + 1
plt.plot([x1, x2], [f(x1), f(x2)], color="k", linestyle="dashed")

plt.show()
