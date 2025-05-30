from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# # reshape() : 1차원 배열을 2차원 배열로 변환
# # np.arange(10) : 0부터 9까지의 정수로 이루어진 1차원 배열 생성
# # values.reshape(-1, 1) : -1은 자동으로 행의 개수를 결정, 1은 열의 개수
# values = np.arange(10).reshape(-1, 1)  

# # fit() : 현재 데이터셋에 평균, 분산, 표준편차 등의 통계적 파라미터를 계산
# # scaler.fit(values) : values 데이터를 사용하여 스케일링 파라미터를 계산
# fit_values = scaler.fit(values)

# 소수점 둘째 자리까지 출력하고, 작은 수를 지수 표기법이 아닌 일반 숫자로 표시
np.set_printoptions(precision=20, suppress=True) 

# 160 * 170 * 190 * 180 = 1,000,800
values1 = np.array((160, 170, 190, 180)).reshape(-1, 1)  # 160 * 170 * 190 * 180 = 1,000,800
# 4,000,000,000,000,000,000 = 4 * 10^18
values2 = np.array((400000000, 700000000, 2000000000, 3000000000)).reshape(-1, 1) 

scaler1 = StandardScaler()
scaler2 = StandardScaler()

# fit_transform() : fit()과 transform()을 한 번에 수행
fit_values1 = scaler1   .fit_transform(values1)
fit_values2 = scaler2.fit_transform(values2)

print("fit_values1:", fit_values1)
print("fit_values2:", fit_values2)

# print("fit_values1:", fit_values1.mean_, fit_values1.var_, fit_values1.scale_)
# print("fit_values2:", fit_values2.mean_, fit_values2.var_, fit_values2.scale_)