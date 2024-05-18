import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('airline-passengers.csv', usecols=[1], engine='python')
plt.plot(dataset)
plt.show()
