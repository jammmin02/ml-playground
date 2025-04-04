import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

np.random.seed(0)  # 랜덤 시드 설정
X= np.random.rand(3, 1) * 10  # 0~10 사이의 랜덤한 값 3개 생성
y = 2.5 * X + np.random.randn(3, 1) * 2  
y = y.ravel()  # 1차원 배열로 변환

from sklearn.linear_model import SGDRegressor

# 모델 생성 후 하이퍼마라미터 설정
model = SGDRegressor(
    # 모델이 데이터를 학습할 최대 반복 횟수 (epoch 수)
    # ex) 데이터 전체를 1000번 반복 학습함
    max_iter=10000,

    # 학습률(learning rate) 조정 방식 설정
    # 'constant' : 학습률을 고정된 값(eta0)으로 유지
    # 'optimal'  : 학습이 진행됨에 따라 자동으로 학습률을 조정
    # 'adaptive' : 일정 횟수 동안 개선이 없으면 학습률을 줄임
    learning_rate='constant',

    # 초기 학습률 값 설정 (learning rate)
    # 값이 너무 크면 발산하고, 너무 작으면 학습이 느려짐
    eta0=0.01,

    # 정규화(Penalty) 사용 여부
    # 'l2', 'l1', 'elasticnet' 등의 옵션이 있으나
    # None 으로 설정하면 정규화를 적용하지 않음
    # → 오로지 손실 함수만 최소화하도록 학습
    penalty=None,

    # 랜덤 시드 고정
    # 결과 재현(reproducibility)을 위해 사용
    # 같은 데이터와 조건에서 같은 결과를 얻을 수 있음
    random_state=0
)

model.fit(X, y)  # 모델 학습

# 평가
# SGDRegressor는 기본적으로 MSE(Mean Squared Error) 손실 함수를 사용
# 모델의 성능 평가를 위해 예측값과 실제값 비교 -> loss(cost) function 사용
# predict() : 모델이 예측한 값
# X : 입력값
y_pred = model.predict(X)  # 예측값

# y_pred : 예측값
# y : 실제값
mse = mean_squared_error(y, y_pred)  # MSE 계산 
