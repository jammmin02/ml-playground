import numpy as np

# H(x) = 5X + 3X + 4
num_of_samples = 5  # 샘플 수
num_of_features = 2  # 특성 수

#data set
np.random.seed(1)  # 랜덤 시드 설정
np.set_printoptions(False, suppress=True)  # 과학적 표기법 사용 안함

X = np.random.rand(num_of_samples, num_of_features) * 10 # 0~10 사이의 랜덤한 값 5개 생성

x_true = [5, 3] # 실제 가중치
b_true = 4 # 실제 절편

noise = np.random.rand(num_of_samples) * 2  # 노이즈 생성 (0~2 사이의 랜덤한 값)

y= X[:, 0] * 5 + X[:, 1] * 3 + b_true + noise  # y = 5X1 + 3X2 + b + noise

print(X)
print()
print(y)