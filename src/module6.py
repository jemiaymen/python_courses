# %%
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
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
df.columns = columns
df = df.drop_duplicates()
df.head(10)
imp_columns = ['age', 
            'fnlwgt', 
            'education-num', 
            'capital-gain', 
            'capital-loss']

data = df[imp_columns]

print(data.isnull().sum())

null_data = data[data.isna().any(axis=1)]

print(null_data)

imputation = SimpleImputer(strategy='mean')
clean_data = imputation.fit_transform(data)
clean_data = pd.DataFrame(clean_data, columns=imp_columns)

print(clean_data.isnull().sum())

# %%
d1 = np.arange(6)
s2 = pd.Series(data=d1, index=['a', 'b', 'c', 'd', 'e', 'f'])
print(s2)

# %%
s2.head(2)
# %%
s2.tail()
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
df = pd.read_csv('../datasets/adult/adult.data', na_values='?')

# %%
df.head(5)
# %%


# %%

df.info()

# %%
df.duplicated().sum()

# %%
df = df.drop_duplicates()
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
columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']
data = df[columns]
data.head(10)
# %%
age_imputation = SimpleImputer(missing_values=np.nan, strategy='mean')


# %%
clean_data = age_imputation.fit_transform(data)
# %%
_clean = pd.DataFrame(clean_data, columns=columns)
_clean.isnull().sum()

# %%
columns = ['workclass',
           'education',
           'marital-status',
           'occupation',
           'relationship',
           'race',
           'sex',
           'native-country',
           'result']
data = df[columns]

print(data.isnull().sum())

age_imputation = SimpleImputer(strategy='most_frequent')
clean_data = age_imputation.fit_transform(data)
clean_data = pd.DataFrame(clean_data, columns=columns)

print(clean_data.isnull().sum())

# %%

columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']
data = df[columns]

print(data.isnull().sum())

imputation = IterativeImputer(max_iter=10, random_state=0)
clean_data = np.round(imputation.fit_transform(data))
clean_data = pd.DataFrame(clean_data, columns=columns)

print(clean_data.isnull().sum())
# %%


columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss']
data = df[columns]

print(data.isnull().sum())

imputation = KNNImputer(n_neighbors=2, weights="uniform")
clean_data = imputation.fit_transform(data)
clean_data = pd.DataFrame(clean_data, columns=columns)

print(clean_data.isnull().sum())

# %%

sub_df = df[['age']]
type(sub_df)

# %%
col = df['age']

type(col)

# %%
sub_df = df[['age', 'workclass', 'education']]
sub_df.head()
# %%
row = df.iloc[1]
row
# %%
rows = df.iloc[0:10]
rows.head(11)

# %%
all_men = df[df['sex'].str.strip() == "Male"]
all_men.tail()

# %%
df.iloc[0]['sex']
# %%
d = {
    'Matier': ['SAS', 'BigData', 'BA'],
    'Score': [10.2, 11.0, 12.5]
}

df = pd.DataFrame(d, index=['code_sas', 'code_bigdata', 'code_ba'])
row = df.loc['code_bigdata']
row
# %%
_all = df[(df['sex'].str.strip() == "Male") & (
    df['education'].str.strip() == "Masters")]
_all.tail()
