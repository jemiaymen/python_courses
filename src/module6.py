# %%
import pandas as pd
import numpy as np

d = {'a': 0, 'b': 1, 'c': 3, 'd': 4}

s = pd.Series(data=d, index=['a', 'b', 'c', 'd'])

print(s)
# %%
s1 = pd.Series(data=d, index=['y', 'z', 'v', 'w'])
print(s1)

# %%
d1 = np.arange(6)
s2 = pd.Series(data=d1, index=['a', 'b', 'c', 'd', 'e', 'f'])
print(s2)

# %%
s2.head(2)
# %%
s2.tail(2)
# %%

data = {
    'Matier': ['SAS', 'BigData', 'BA'],
    'Score': [10.2, 11.0, 12.5]
}

df = pd.DataFrame(data)

# %%
df
# %%
df = pd.DataFrame(data, index=['code_sas', 'code_bigdata', 'code_ba'])


# %%
df
# %%
sas = df.loc['code_sas']

# %%
sas
# %%
d1 = np.arange(6)
d2 = np.arange(6, 12)

index = ['a', 'b', 'c', 'd', 'e', 'f']
s1 = pd.Series(data=d1, index=index)
s2 = pd.Series(data=d2, index=index)

df = pd.DataFrame([s1, s2])

# %%
df
# %%
df.index = ['MA', 'ME']

# %%
df
# %%
df = pd.read_csv('../datasets/adult/adult.data',na_values='?')

# %%
df.head(5)
# %%

columns = ['age',
         'workclass',
         'fnlwgt',
         'education',
         'education-num',
         'marital-status',
         'occupation',
         'relationship',
         'race',
         'sex',
         'capital-gain',
         'capital-loss',
         'hours-per-week',
         'native-country',
         'result']

df = pd.read_csv('../datasets/adult/adult_custom.data',
                 header=None, na_values=' ?')

df.head(5)
df.columns = columns

df.head(5)
# %%

df.info()

# %%
df.duplicated().sum()
# %%

df.isnull().sum()
# %%
null_data = df[df.isna().any(axis=1)]
null_data
# %%
null_data.shape
# %%
df = df.dropna()

# %%
df.isna().sum()

# %%
