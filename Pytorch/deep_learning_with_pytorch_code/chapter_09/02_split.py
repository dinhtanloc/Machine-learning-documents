import numpy as np
data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# find the boundary at 66% of total samples
count = len(data)
n_train = int(count * 0.66)
# split the data at the boundary
train_data = data[:n_train]
test_data = data[n_train:]
