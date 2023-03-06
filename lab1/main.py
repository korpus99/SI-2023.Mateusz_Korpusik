import pandas as pd
import numpy as np
from scipy.stats import mode, zscore

# Zadanie2
attribute_list =  ["a1n", "a2n", "a3n", "a4n", "a5n", "a6n", "a7n", "a8n", "a9d"]
dataframe= pd.read_csv("dane/diabetes.txt", sep=" ", header=None)
info = pd.read_csv("dane/_info-data-discrete.txt", sep=" ", header=None)

print(dataframe)

# Zadanie3
# a
print("\nSymbole klas decyzyjnych:\n",pd.unique(dataframe[1]))
# b
print("\nWielkoś klas decyzyjnych:\n" ,info[2][9])
# c
print("\nMinimalna wartość atrybutów:\n")
for col in dataframe.columns:
    print(f"{col} : {dataframe[col].min()}")
print("\nWartosci maksymalna atrybutów:\n")
for col in dataframe:
    print(col, ":", dataframe[col].max())
# d
print("\nLiczba różnych wartości:\n")
for index, col in enumerate(dataframe.columns):
    print(f"{attribute_list[index]}: {len(dataframe[col].unique())}")
# e
print("\nLista dostępnych różnych wartości:\n")
for index, col in enumerate(dataframe.columns):
    print(f"{attribute_list[index]}: {list(dataframe[col].unique())}")
# f
print("\nOdchylenie standardowe:\n")
for col in range(0, 7):
    print(f"{attribute_list[col]} : {dataframe[col].std()}")

# Zadanie4
# a
data = np.random.rand(100, 5) * 10
data[:10, :] = np.nan
print(data[:15, :])

missing_ratio = 0.1
missing_mask = np.random.choice([True, False], size=data.shape, p=[missing_ratio, 1 - missing_ratio])

for i in range(data.shape[1]):
    col = data[:, i]
    mode_val = mode(col, nan_policy='omit', axis=0, keepdims=True)[0][0]
    col[np.isnan(col)] = mode_val
    data[:, i] = col

print(data[:15, :])

# b
data = np.random.rand(100, 5) * 10

a = -1
b = 1
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

a = 0
b = 1
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

a = -10
b = 10
norm_data = (((data - np.min(data, axis=0)) * (b - a)) / (np.max(data, axis=0) - np.min(data, axis=0))) + a
print(norm_data[:10, :])

# c
print("\nPrzed Standaryzacją danych")
data_scaled = pd.DataFrame(zscore(dataframe), columns=dataframe.columns)
print(data_scaled.describe())
print(data_scaled.var())
print("\nPo Standaryzacji danych")
print(data_scaled)

# d
data = pd.read_csv('dane/Churn_Modelling.csv')

data = data.drop(['RowNumber', 'CustomerId', 'Age'], axis=1)

geography_dummies = pd.get_dummies(data['Geography'], drop_first=True)
data = pd.concat([data, geography_dummies], axis=1)
data = data.drop(['Geography'], axis=1)

print(data.head())
