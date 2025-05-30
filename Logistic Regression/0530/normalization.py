from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


x = np.arange(1, 11)

max = x.max()
min = x.min()

values = [(item - min) / (max - min) for item in x]
print("values:", values)
