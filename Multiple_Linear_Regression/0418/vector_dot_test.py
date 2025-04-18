import numpy as np

# seed : 값을 고정시켜서 랜덤한 값을 생성할 때 매번 같은 값이 나오도록 하는 것
np.random.seed(1)

x = np.random.randint(1, 4, (2, )) # 1차원 배열을 생성
y = np.random.randint(1, 4, (2, )) # 2차원 배열을 생성

print(f"x: {x} y: {y}\n {x + 2}\n {x - 2}\n {x * 2}\n {x / 2}")

# x @ y : 행렬의 내적을 구하는 연산자 = 스칼라(수치적인 값)
# 내적 : 두 벡터의 곱을 구하는 연산 
# x + y : 두 벡터의 합을 구하는 연산 = 백터(힘과 방향)

x = np.array([[1, 2], [3, 4]]) # 2차원 배열을 생성
w = np.array([[2], [3]]) # 2차원 배열을 생성

#shape : 배열의 차원과 크기를 나타내는 속성
print(x.shape) # (2, 2) : 2행 2열
print(w.shape) # (2, 1) : 2행 1열