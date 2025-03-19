import numpy as np
import matplotlib.pyplot as plt

# data set
# input : input data, feature
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand()  * 10 for _ in range(50)]

# output : label
# f(x) -> f(x2)
# y = label 
y_train = [val + np.random.rand() * 5 for val in x_train]

# BGC(Batch Gradient Descent) : 배치 경사 하강법
# 이용하여 선형 회귀 모델(Linear Rergeresion)을 구현 = Mean Squared Error

plt.scatter(x_train, y_train, color='pink')
plt.show()