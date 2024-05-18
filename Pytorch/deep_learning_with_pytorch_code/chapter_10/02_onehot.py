import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("iris.csv", header=None)
X = data.iloc[:, 0:4]
y = data.iloc[:, 4:]

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
print(ohe.categories_)

y = ohe.transform(y)
print(y)
