import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


a = np.random.standard_normal(size=10)

print(a)

a = np.where(a > 0.4, a - 0.4, np.where(a < -0.4, a + 0.4, 0))

print(a)


df = pd.read_csv("data/Sonar.csv")

minmax_df = MinMaxScaler().fit_transform(df[:, :-1])

