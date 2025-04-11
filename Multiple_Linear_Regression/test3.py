sample = []
y = []

w = [0.2, 0,3]
b = 0.1

gradient_w = [0.0, 0.0]
gradient_b = 0.0

# 모든 샘플 순회 : 1 epoch
for dp, y_ in sample:
  
  # 예측값
  predict_y = w[0] * dp[0] + w[1] * dp[1] + b
  
  # 오차 : 예측값 - 실제값
  error = predict_y - y_

  # 기울기값 누적
  gradient_w[0] += dp[0] * error
  gradient_w[1] += dp[1] * error
  
# update gradient of each w
# update gradient of b
w[0] = w[0] - gradient_w[0] / len(sample)
w[1] = w[1] - gradient_w[1] / len(sample)
gradient_b += error
