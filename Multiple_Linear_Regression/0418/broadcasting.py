import numpy as np

# seed : 값을 고정시켜서 랜덤한 값을 생성할 때 매번 같은 값이 나오도록 하는 것
np.random.seed(5)

x = np.random.randint(1, 4, (2, 2)) # 1차원 배열을 생성
y = np.random.randint(1, 4, (2, 1)) # 2차원 배열을 생성

#  브로드캐스트 : 서로 다른 차원의 배열을 연산할 수 있도록 해주는 것
# 조건 1 : 차원이 같아야 한다.
# 조건 2 : 하나가 1이여야 한다
print(f"x: {x} \n\n y: {y} \n\n {x + y}")