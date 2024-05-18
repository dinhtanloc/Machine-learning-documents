import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# load dataset
data = pd.read_csv("sonar.csv", header=None)
# split into input (X) and output (Y) variables, in numpy arrays
X = data.iloc[:, 0:60].values
y = data.iloc[:, 60].values

# binary encoding of labels
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# create model
model = MLPClassifier(hidden_layer_sizes=(60,60,60), activation='relu',
                      max_iter=150, batch_size=10, verbose=False)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print("mean = %.3f; std = %.3f" % (results.mean(), results.std()))
