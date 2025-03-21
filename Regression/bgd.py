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

w = 0.0 # 초기 가중치
b = 0.0 # 초기 절편
learning_rate = 0.01 # 학습률
epoch = 100  # 반복 횟수(학습률)
loss_history = [] # 손실값 저장


for num_of in range(epoch):
    gradient_w_sum = 0.0 # 기울기 합
    gradient_b_sum = 0.0 # 절편 합
    loss = 0.0 # 손실값
    
    # GD 수행 후 최적의 w를 찾아야 함
    # zip(list1, list2) : list1, list2를 묶어줌
    for x, y in zip(x_train, y_train):
        gradient_w_sum += x * (w * x  + b - y) # 기울기 합
        gradient_b_sum += (w * x + b - y) # 절편 합
        
        loss += (w * x + b - y) ** 2 # 손실값
    
    # w값 업데이트
    # w = w - learning_rate * gradient_sum
    w = w - learning_rate * (gradient_w_sum / len(x_train))
    b = b - learning_rate * (gradient_b_sum / len(x_train))
    
    loss_history.append(loss / len(x_train)) # 손실값 저장
    
  
# 테스트 데이터
x_test = [val for val in range(10)]
y_test = [w * val + b for val in x_test]


# 시각화
fig, axe = plt.subplots(1, 2) # 1행 2열
axe[0].set_title('Linear Regression') # 제목
axe[0].scatter(x_train, y_train, color='green') # 학습 데이터
axe[0].set_xlabel('x') # x축
axe[0].set_ylabel('y') # y축

axe[1].set_title('Loss History') # 제목
axe[0].plot(x_test, y_test, color='blue') # 예측값
axe[1].plot(loss_history, color='red') # 손실값
axe[1].set_xlabel('Epoch') # x축
axe[1].set_ylabel('Loss') # y축

plt.show()
  