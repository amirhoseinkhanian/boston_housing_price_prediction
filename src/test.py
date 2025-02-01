import pandas as pd

data = pd.read_csv("data/BostonHousing.csv")
print(data.isna().sum())
print(data.isnull().sum().sum())
