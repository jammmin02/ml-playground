from sklearn.preprocessing import StandardScaler
import numpy as np


x = np.arange(10)

x_sum = sum(x)
x_avg = x_sum / len(x)
x_variance = 0.0

# 분산 계산 (Variance)
# 분산(Variance) = (각 데이터 - 평균)^2의 합 / 데이터 개수
for item in x:
    x_variance += (item - x_avg) ** 2

# 분산을 데이터 개수로 나누기
x_variance /= len(x)
print("Variance:", x_variance)

# 분산 계산 (Numpy)
np_avg = x.mean() 

# var() -> numpy에서 분산을 계산하는 함수
np_variance = np.var(x)
print("Numpy Variance:", np_variance)

# 표준편차 계산 (Standard Deviation)
# std() -> numpy에서 표준편차를 계산하는 함수
np_std = np.std(x)
print("Standard Deviation:", np_std)