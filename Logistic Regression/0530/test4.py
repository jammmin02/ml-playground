import numpy as np

one_hot = np.eye(4)

print(one_hot, "\n\n")

my_list = [0, 1, 0, 3, 2, 3]

one_hot_list = one_hot[my_list]

print(one_hot_list)
print(np.argmax(one_hot_list, axis=1))