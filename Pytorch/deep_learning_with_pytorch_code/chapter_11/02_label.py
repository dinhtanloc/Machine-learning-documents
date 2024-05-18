import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read data
data = pd.read_csv("sonar.csv", header=None)
X = data.iloc[:, 0:60]
y = data.iloc[:, 60]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
