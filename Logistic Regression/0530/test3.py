from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 표준편차를 사용하여 각 데이터에서 평균을 빼기
# 평균과 표준편차는 학습 데이터에서 계산 (fit)
# 테스트 데이터는 같은 기준으로 변환해야 하므로 평균/표준편차 고정 필요
# 테스트 시엔 이미 계산된 값 사용 → transform만 수행 (fit 불필요)

values = [ item - mean for item in x ]

x = np.arange(10)

print(x.mean(), x.std())

mean = x.mean()

values = [ item - mean for item in x ]
print("values:", values)
