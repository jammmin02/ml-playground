import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 데이터 생성
x_train = np.array([np.random.rand() * 10 for _ in range(50)])
y_train = np.array([x + np.random.rand() * 5 for x in x_train])

# -----------------------------
# 초기 설정
w = 0.0
b = 0.0
learning_rate = 0.01
epoch = 100
history = []       # (w, b)
loss_history = []  # 손실값 기록

# -----------------------------
# 학습 & 기록 저장
for _ in range(epoch):
    dw_sum = 0.0
    db_sum = 0.0
    loss = 0.0

    for x, y in zip(x_train, y_train):
        y_pred = w * x + b
        error = y_pred - y
        dw_sum += error * x
        db_sum += error
        loss += error ** 2

    w -= learning_rate * (dw_sum / len(x_train))
    b -= learning_rate * (db_sum / len(x_train))

    history.append((w, b))
    loss_history.append(loss / len(x_train))  # 평균 제곱 오차 (MSE)

# -----------------------------
# 시각화 준비
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_test = np.linspace(0, 10, 100)

# 왼쪽: 회귀선
axes[0].scatter(x_train, y_train, color='green', label='Train Data')
line, = axes[0].plot([], [], color='blue', label='Prediction Line')
axes[0].set_xlim(0, 10)
axes[0].set_ylim(min(y_train) - 5, max(y_train) + 5)
axes[0].set_title('Linear Regression')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()

# 오른쪽: 손실값
loss_line, = axes[1].plot([], [], color='red', label='Loss')
axes[1].set_xlim(0, epoch)
axes[1].set_ylim(0, max(loss_history) + 5)
axes[1].set_title('Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

# -----------------------------
# 애니메이션 프레임 함수
def update(frame):
    w, b = history[frame]
    y_pred = w * x_test + b
    line.set_data(x_test, y_pred)

    # 손실값 그래프
    loss_line.set_data(range(frame + 1), loss_history[:frame + 1])

    # 제목 업데이트
    axes[0].set_title(f'Linear Regression (Epoch {frame + 1})')
    axes[1].set_title(f'Loss Over Epochs (Epoch {frame + 1})')

    return line, loss_line

# -----------------------------
# 애니메이션 실행
ani = FuncAnimation(fig, update, frames=epoch, interval=100, blit=True)

plt.tight_layout()
plt.show()
