import numpy as np

# h(x) = wx1 + wx2 + wx3 + b
# 차원 확인인
# kin = np.array(1)
# bar = np.array([1, 2, 3])
# forr  =  np.array([[1], [2], [3]])

# print(f", {kin.shape}, {bar.shape}, {forr.shape}")
# print()
# print(f"{type(bar)}, {type(forr)}, {type(kin)}")
# ---------------------------------------------------------------------------------

# h(x) = wx1 + wx2 + wx3 + b
# 초기값 설정
kin = np.ones((2, 3, 4))
bar = np.zeros((5, 2))

print(f", {kin.shape}, {bar.shape}")
print()
print(f"{kin}, {bar}")
