#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# load dataset
data = pd.read_csv('car_prediction.txt')
df = data.copy()
data.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.isna().sum()


# In[6]:


df.info()
# All the features are in object


# In[7]:


df['year'].value_counts().index


# In[8]:


# CLEANING THE 'YEAR' COLUMN

# The year column contains other string values which has no year menationed - irrelevant data 
# removing all those years and converting the year column to numeric
df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)


# In[9]:


df['Price'].value_counts().index


# In[10]:


# CLEANING THE PRICE COLUMN

# The price columns contains some strings and also the values are in commas
# removing all the strings and converting into numeric

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(',','').astype(int)


# In[11]:


df['kms_driven'].value_counts()


# In[12]:


#CLEANING THE KMS_DRIVEN COLUMN

# The column contains commas in values and kms in the end

df['kms_driven'] = df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
df = df[df['kms_driven'].str.isnumeric()]
df['kms_driven'] = df['kms_driven'].astype(int)


# In[13]:


df['fuel_type'].value_counts()


# In[14]:


# CLEANING FUEL_TYPE

# There is one NaN value 
# convert the categories to numeric- label encoding

df = df[~df['fuel_type'].isna()]


# In[15]:


# CLEANING NAME column

# Keep only the first 3 words of the name

df['name'] = df['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[16]:


# CLEANED DATA
df.reset_index(drop = True)


# In[17]:


df.info()


# In[18]:


df.describe()


# In[19]:


# In the price the max value is high

df = df[df['Price']<6e6].reset_index(drop = True)


# In[20]:


df.to_csv('Cleaned_dataset_car.csv', index = False)


# In[21]:


data_cleaned = pd.read_csv('Cleaned_dataset_car.csv')


# In[22]:


data_cleaned


# In[23]:


# EDA
sns.pairplot(data_cleaned)


# In[24]:


sns.distplot(data_cleaned['Price'])


# In[25]:


sns.heatmap(data_cleaned.corr(), annot = True)


# In[26]:


X = data_cleaned.drop(columns = 'Price')
Y = data_cleaned['Price']


# In[27]:


print(X.shape, Y.shape)


# In[28]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[30]:


ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])


# In[31]:


column_trans = make_column_transformer ((OneHotEncoder(categories = ohe.categories_),['name', 'company', 'fuel_type']),
                                       remainder = 'passthrough')


# In[32]:


lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(X_train, Y_train)

lr.intercept_, lr.coef_

y_pred_test = pipe.predict(X_test)
y_pred_train = pipe.predict(X_train)


# In[33]:


r2_score(Y_test,y_pred_test)


# In[34]:


r2_score(Y_train,y_pred_train)

# OVERFITTING


# In[35]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test,y_pred_test))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test,y_pred_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test,y_pred_test)))


# In[36]:


data_cleaned


# In[37]:


import pickle


# In[38]:


pickle.dump(pipe, open('LinearRegressionModel.pkl','wb'))


# In[39]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2001,200,'Petrol']],
                         columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type']))


# In[40]:


data_cleaned[(data_cleaned['company']=='Maruti') & (data_cleaned['year']==2019)]


# In[ ]:





# In[ ]:




