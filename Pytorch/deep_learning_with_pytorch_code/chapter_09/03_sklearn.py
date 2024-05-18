import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
train_data, test_data = train_test_split(data, test_size=0.33)
