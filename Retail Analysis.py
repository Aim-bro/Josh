## 데이터 불러오기
# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")


# %%


# CSV 파일 읽기
df = pd.read_csv(r'C:\Users\Jo\python\BOJ\Dataset\file_out2.csv', parse_dates=['Date'])

# 데이터 확인
print(df.head())  # 상위 5행 확인
print(df.shape)   # 데이터 크기 확인
print(df.columns) # 컬럼명 확인

# %%

category_vars = ['Unnamed: 0', 'InvoiceID', 'ProductID', 'CustomerID']

for _ in category_vars:
    df[_] = df[_].astype('object')

# df.describe()
df.info()
# %%
df.nunique()
# %%
df.drop('Unnamed: 0', axis=1, inplace=True)
df.drop_duplicates(keep='first', inplace=True)


# %%
df.isnull().sum()

# %%
(df[['TotalSales', 'Quantity']] == 0).sum()
# %%
