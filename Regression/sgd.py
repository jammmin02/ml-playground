import numpy as np
import matplotlib.pyplot as plt
import random

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

w = 0.0 # 초기 가중치
learning_rate = 0.001 # 학습률
epoch = 50  # 반복 횟수(학습률)
loss_history = [] # 손실값 저장

# SGD 
# H(x) -> w * x + b
for num_of in range(epoch):
  
    data = list(zip(x_train, y_train)) # 데이터 묶음
    random.shuffle(data) # 데이터 랜덤으로 섞음
    
    # GD 수행 후 최적의 w를 찾아야 함
    for x, y in data:
        # w값 업데이트
        w = w - learning_rate * (x * (w * x - y)) # w 값 업데이트

# 테스트 데이터
x_test = [val for val in range(10)]
y_test = [w * val for val in x_test]

# 시각화
plt.title('Linear Regression') # 제목
plt.scatter(x_train, y_train, color='green') # 학습 데이터
plt.plot(x_test, y_test, color='blue') # 예측값
plt.xlabel('x') # x축
plt.ylabel('y') # y축
plt.show()
  
  