import pandas as pd
import numpy as np

df = pd.read_csv("Dataset_Open_Items_EN.csv", sep=';')
print(df.columns)
print(df.head(2))
#print(df['amount'])
print(df['Payment_delay'])
dtype = df.dtypes
print(dtype)
floats = df.select_dtypes(include=[np.float64])
print(floats)