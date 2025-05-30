from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# randn : 표준 정규 분포에서 난수 생성
# 난수 발생 범위가 필요 없는 이유
# np.random.randn() 함수는 표준 정규분포(평균 0, 표준편차 1)를 따르는 난수를 생성
# 이 함수는 특정 구간 내에서 난수를 생성하는 것이 아니라, 정규분포의 특성상 이론적으로 -∞부터 +∞까지의 값을 가질 수 있음
# 즉, 값의 범위를 지정할 필요 없이 정규분포 자체에서 난수를 추출하기 때문에 별도의 범위 지정이 필요 없음
# np.random.randn() : 평균이 0이고 표준편차가 1인 정규분포에서 난수를 생성
values = [np.random.randn() for _ in range(10000)]

# np.random.randn() * 10 + 50 : 평균이 50이고 표준편차가 10인 정규분포에서 난수 생성
values_2 = [np.random.randn() * 10 + 50 for _ in range(10000)]

