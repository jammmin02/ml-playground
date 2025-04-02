import numpy as np
import random
import matplotlib.pyplot as plt


# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]

# BGD
# 1. H(x) = W * x

# 2. optimizer (W값을 업데이트 함)[GD] : W = W - slope of cost function for given W
# 2. 옵티마이저 : W = W - cost function의 기울기
# BGD : w = w - 
# SGD : w = w - 