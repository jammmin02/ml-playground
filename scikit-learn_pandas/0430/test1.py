import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 데이터로딩
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

# 3. 입력 특성 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 모델 훈련
model = SGDRegressor(
  max_iter=1000,
  tol=0.01,
  eta0=0.001,
  learning_rate='constant',
  penalty='l1',
  random_state=42
)
model.fit(X_train_scaled, y_train)

# 5. 예측
y_pred = model.predict(X_test_scaled)

# MSE와 R^2 점수 계산
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred):.4f}")

# 6. 회귀계수출력
print("\n회귀계수 (weights):")
for name, coef in zip(feature_names, model.coef_):
    print(f"{name:<20}: {coef:20,.2f}")

# 6. 절편 출력
print(f"\n절편 (intercept): {model.intercept_[0]:.2f}")

# 7. 예측 결과 시각화
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("실제값 (y_test)")
plt.ylabel("예측값 (y_pred)")
plt.title("실제값 vs 예측값")
plt.grid(True)
plt.show()