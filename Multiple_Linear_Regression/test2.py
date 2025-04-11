# 입력 데이터 (독립 변수: x값)
sample = []  # 예: [1.0, 2.0, 3.0]

# 정답 데이터 (종속 변수: y값)
y = []       # 예: [2.0, 4.0, 6.0]

# 초기 가중치(w)와 바이어스스(b) 설정
w = 0.2      # 직선의 기울기 초기값
b = 0.1      # 직선의 y절편 초기값

# 기울기 누적 변수 초기화 (gradient_w: w 방향, gradient_b: b 방향)
gradient_w = 0.0
gradient_b = 0.0

# 입력 데이터와 정답 데이터를 쌍으로 묶어서 반복
for f, y_ in zip(sample, y):
    # 예측값 계산: H(x) = w * x + b
    predict_y = w * f + b

    # 오차(error): 예측값 - 실제값
    error = predict_y - y_

    # 가중치 w에 대한 기울기 누적 (∂L/∂w = error * x)
    gradient_w += error * f

    # 바이어스 b에 대한 기울기 누적 (∂L/∂b = error)
    gradient_b += error

# 평균 기울기만큼 w와 b를 경사 하강법으로 업데이트
w = w - gradient_w / len(sample)
b = b - gradient_b / len(sample)
