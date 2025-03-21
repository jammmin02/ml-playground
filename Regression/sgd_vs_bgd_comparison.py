import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 데이터 생성
np.random.seed(42)  # 랜덤 고정 (재현 가능)
x_train = np.random.rand(50) * 10
y_train = 2 * x_train + 3 + np.random.randn(50) * 2  # y = 2x + 3 + noise

# -----------------------------
# 초기값 설정
w_sgd, w_bgd = 0.0, 0.0  # SGD & BGD 가중치 초기값
b_sgd, b_bgd = 0.0, 0.0  # 절편 초기값
learning_rate = 0.01
epochs = 100

# 기록 저장용 리스트
w_sgd_history, w_bgd_history = [], []
loss_sgd_history, loss_bgd_history = [], []

# -----------------------------
# BGD (배치 경사 하강법)
for epoch in range(epochs):
    y_pred_bgd = w_bgd * x_train + b_bgd
    error_bgd = y_pred_bgd - y_train
    
    dw_bgd = np.mean(error_bgd * x_train)
    db_bgd = np.mean(error_bgd)
    
    w_bgd -= learning_rate * dw_bgd
    b_bgd -= learning_rate * db_bgd
    
    loss_bgd = np.mean(error_bgd ** 2)  # MSE
    w_bgd_history.append(w_bgd)
    loss_bgd_history.append(loss_bgd)

# -----------------------------
# SGD (확률적 경사 하강법)
for epoch in range(epochs):
    rand_idx = np.random.randint(0, len(x_train))  # 랜덤 샘플 하나 선택
    x_sgd, y_sgd = x_train[rand_idx], y_train[rand_idx]
    
    y_pred_sgd = w_sgd * x_sgd + b_sgd
    error_sgd = y_pred_sgd - y_sgd
    
    dw_sgd = error_sgd * x_sgd
    db_sgd = error_sgd
    
    w_sgd -= learning_rate * dw_sgd
    b_sgd -= learning_rate * db_sgd
    
    loss_sgd = error_sgd ** 2  # 개별 샘플에 대한 손실
    w_sgd_history.append(w_sgd)
    loss_sgd_history.append(loss_sgd)

# -----------------------------
# 그래프 그리기
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽 그래프: W 값 변화
axes[0].plot(w_sgd_history, 'b--', label='SGD W 변화')  # 파란색 점선
axes[0].plot(w_bgd_history, 'r-', label='BGD W 변화')  # 빨간색 실선
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('W 값')
axes[0].set_title('SGD vs BGD W 값 변화')
axes[0].legend()

# 오른쪽 그래프: Loss 값 변화
axes[1].plot(loss_sgd_history, 'b--', label='SGD Loss 변화')  # 파란색 점선
axes[1].plot(loss_bgd_history, 'r-', label='BGD Loss 변화')  # 빨간색 실선
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss 값')
axes[1].set_title('SGD vs BGD Loss 변화')
axes[1].legend()

plt.tight_layout()
plt.show()
