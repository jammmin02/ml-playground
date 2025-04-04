import numpy as np

# numpy는 파이썬에서 수치 계산을 위한 라이브러리

# zeros() : 주어진 shape의 배열을 생성하고 모든 원소를 0으로 초기화
bar = np.zeros((2))
foo = np.zeros((3, 2))
pos = np.zeros((2, 3, 2))

# shape란?
# shape는 배열의 차원과 크기를 나타내는 속성
# 배열의 각 차원에 대한 크기를 튜플 형태로 반환
# 예를 들어, (2, 3) 형태의 배열은 2행 3열의 배열을 의미

# (2,)는 1차원 배열로 2개의 원소를 가진 배열을 의미 ->  백터
# (3, 2)는 2차원 배열로 3행 2열의 배열을 의미 -> 메트릭스
# (2, 3, 2)는 3차원 배열로 2개의 3행 2열의 배열을 의미 -> 텐서

print(f"bar.shape: {bar.shape}")  # (2,)
print(f"foo.shape: {foo.shape}")  # (3, 2)
print(f"pos.shape: {pos.shape}")  # (2, 3, 2)
print()

print(bar) # [0. 0.] -> 1차원 배열
print() 
print(foo) # [[0. 0.] [0. 0.] [0. 0.]] -> 2차원 배열
print()
print(pos) # [[[0. 0.] [0. 0.] [0. 0.]] [[0. 0.] [0. 0.] [0. 0.]]] -> 3차원 배열