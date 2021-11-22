# %%
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

#%%
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

imp_columns = ['age',
               'fnlwgt',
               'education-num',
               'capital-gain',
               'capital-loss']

df = pd.read_csv('../datasets/adult/adult_custom.data', header=None, na_values=' ?')
df2 = pd.read_csv('../datasets/adult/adult.data', header=None, na_values=' ?')
df2 = df2.drop_duplicates()
df = df.drop_duplicates()
df.columns = columns
df2.columns = columns


imp_data = df[imp_columns]
null_data = imp_data[imp_data.isna().any(axis=1)]
null_indexs = null_data.index

df2 = df2[imp_columns]
clean_data = df2.iloc[null_indexs]

#%%
clean_data.head(6)
#%%
null_data.head(6)
#%%
# mean imputation

imputation = SimpleImputer(strategy='mean')
clean_data_mean = imputation.fit_transform(imp_data)
clean_data_mean = pd.DataFrame(clean_data_mean, columns=imp_columns)

imputed_with_mean = clean_data_mean.iloc[null_indexs]
#%%
print('imputer with mean')
imputed_with_mean.head(6)
#%%
print('clean data')
clean_data.head(6)

#%%
# median imputation

imputation = SimpleImputer(strategy='median')
clean_data_median = imputation.fit_transform(imp_data)
clean_data_median = pd.DataFrame(clean_data_median, columns=imp_columns)

imputed_with_median = clean_data_median.iloc[null_indexs]
#%%
print('imputer with median')
imputed_with_median.head(6)
#%%
print('clean data')
clean_data.head(6)

#%%
# iterativ imputation
imputation = IterativeImputer(max_iter=10, random_state=0)
clean_data_iterativ = np.round(imputation.fit_transform(imp_data))
clean_data_iterativ = pd.DataFrame(clean_data_iterativ, columns=imp_columns)

imputed_with_iterativ = clean_data_iterativ.iloc[null_indexs]
#%%
print('imputer with iterativ')
imputed_with_iterativ.head(6)
#%%
print('clean data')
clean_data.head(6)

# %%
null_data.head(6)
# %%

d = {
    'Matier': ['SAS', 'BigData', 'BA'],
    'Score': [10.2, 11.0, 12.5]
}

df = pd.DataFrame(d, index=['code_sas', 'code_bigdata', 'code_ba'])
row = df.iloc[1]

# %%
df[df['Matier'] == 'BA']
# %%
codes, uniques = pd.factorize(df['education'])
print(codes)
print(uniques)

# %%
codes
# %%

# plot a simple Histogram based on a single column
df['age'].plot(kind='hist', title='Age')
# %%
df_toplot = pd.DataFrame({'lab' : ['A','B','C'] , 'val' : [20, 10, 30]})
df_toplot.plot(kind='bar' , x='lab' , y='val' , rot=0)
# %%
df_toplot.plot(kind='pie' , x='lab' , y='val' , rot=0)
# %%
ages = [20,22,29,21,22]
scores = [10.3, 13.2 ,11.0, 15.2, 13.0 ]
names = ['Aymen','Jemi' , 'Foulane' , 'Man', 'Woman']
df_toplot = pd.DataFrame({'age' : ages,'score' : scores} , index=names)
df_toplot.plot(kind='bar' , rot=0)

# plot scatter
df.plot(kind='scatter', x='age', y='hours-per-week' , title='Age By Hours Work')

# plot box
#%%
df_toplot.plot(kind='box' , title='INAE')
# %%
