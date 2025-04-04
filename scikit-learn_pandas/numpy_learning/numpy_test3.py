import numpy as np

np.set_printoptions(suppress=True, precision=1) # 소수점 이하 1자리까지 출력

# X = 3개의 샘플(sample)과 1개의 특성(feature)을 가진 입력 데이터 (3행 1열)
X = np.random.rand(3, 1) * 10 

# H(x) = w * x + b
# y = 2.5 * X + np.random.randn(100, 1)

pos = 2.5 * X 

# bar : 정규분포를 따르는 랜덤한 값 (3행 1열)
bar = np.random.randn(3, 1) * 2 # -2 ~ 2 사이의 랜덤한 값 추가

# y : 출력값 (정답값)
y = 2.5 * X + bar 

print(X)
print("---" * 10)
print(pos)
print("---" * 10)
print(bar)
print("---" * 10)
print(y)
print("---" * 10)

import numpy as np
import matplotlib.pyplot as plt

# 보기 좋은 출력 설정
np.set_printoptions(suppress=True, precision=1)

# 데이터 생성
X = np.random.rand(3, 1) * 10
pos = 2.5 * X
bar = np.random.randn(3, 1) * 2
y = pos + bar

# 시각화
plt.figure(figsize=(8, 6))

# X 값 그대로, pos (선형 예측값): 파란색 점
plt.scatter(X, pos, color='blue', label='Ideal (2.5 * X)', s=100, marker='o')

# X 값 그대로, y (노이즈 포함 실제값): 빨간 점
plt.scatter(X, y, color='red', label='Actual (2.5 * X + noise)', s=100, marker='x')

# pos → y로 향하는 선 (노이즈 벡터 시각화)
for i in range(len(X)):
    plt.plot([X[i], X[i]], [pos[i], y[i]], color='gray', linestyle='dotted')

# 그래프 꾸미기
plt.title("Linear Relationship with Noise", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
