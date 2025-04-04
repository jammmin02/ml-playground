import numpy as np

# <class 'numpy.ndarray'> -> numpy의 배열 객체
# rnadom.rand() : 0~1 사이의 랜덤한 값을 생성
# rand() : 차원을 의미함함
print(type(np.random.rand(2, 3)))


# set_printoptions : numpy 배열의 출력 형식 설정
# suppress : 지수 표기법을 사용하지 않음 (기본값 = False)
# precision : 소수점 이하 자릿수 (기본값 = 8)
np.set_printoptions(suppress = True, precision = 2) 

# 2행 3열의 랜덤한 값 생성
bar = np.random.rand(2, 3) 

print(bar) # 2차원 배열
print("----" * 10)

bar = bar * 10 # 0~10 사이의 랜덤한 값 생성
print(bar) 