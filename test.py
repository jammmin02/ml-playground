from sklearn.model_selection import train_test_split
import numpy as np

# x : 입력값
X = np.random.rand(10, 2) * 5 # 0~5 사이의 랜덤한 값 100개 생성

# y :  출력값 (정답값)
Y = np.random.randint(0, 2, size=10) # 정답값


# train_test_split : 데이터를 학습용과 테스트용으로 나누는 함수
# test_size : 테스트 데이터 비율 (0.8 = 80%)
# random_state : 랜덤 시드 (같은 결과를 얻기 위해 사용)
# stratify : 비율을 맞추기 위해 사용 (default = None)
# stratify = Y : Y값을 기준으로 비율을 맞춤
# shuffle : 데이터를 섞을지 여부 (default = True)
# shuffle = False : 데이터를 섞지 않음
# shuffle = True : 데이터를 섞음

# for i in zip(X, Y):
#   print(i) # X와 Y를 묶어서 출력
  

X_train, X_test, Y_train, Y_test = \
  train_test_split(X, Y, test_size=0.2, random_state=1) # 80%를 테스트 데이터로 사용

print(f"X_train.shape : {X_train.shape}")
print(X_train) 

print(f"Y_train.shape : {Y_train.shape}")
print(Y_train)

