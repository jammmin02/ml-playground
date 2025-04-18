import numpy as np

num_of_samples = 1000  # 샘플 수
num_of_features = 4  # 특성 수

np.random.seed(5)

# 0~2 사이의 랜덤한 값 1000개 생성
X = np.random.rand(num_of_samples, num_of_features) * 2
w_true = np.random.randint(1, 11, (num_of_features, 1))
b_true = np.random.randn() * 0.5  # 절편
y= X @ w_true + b_true

#-------------------------------------------------------------------------------------------

w = np.random.rand(num_of_features, 1) # 가중치
b = np.random.rand() # 절편
learning_rate = 0.01 # 학습률
epoch = 10000 # 반복 횟수

gradient = np.zeros(num_of_features,) # 가중치의 기울기

for _ in range(epoch): # 1000번 반복
  # 예측값
  predict_y = X @ w + b

  # 오차
  error = predict_y - y

  # 기울기
  # X.T : 전치행렬
  # X.T @ error -> X.T와 error의 내적을 구함
  gradient_w = X.T @ error / num_of_samples 
  # 절편의 기울기
  # mean() : 평균값을 구하는 함수
  # error.mean() : error의 평균값을 구함
  gradient_b = error.mean()

  # print (gradient_w.shape) # (4, 1)
  # print (gradient_b.shape) # ()


  # w, b 업데이트
  w = w - gradient_w * learning_rate # 가중치 업데이트
  b = b - gradient_b * learning_rate # 절편 업데이트
  
print(f"w :\n {w} \n")
print(f"b :\n {b} \n")

print(f"w_true :\n {w_true}" )
print(f"b_true :\n {b_true}")