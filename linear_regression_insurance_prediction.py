#!/usr/bin/env python
# coding: utf-8

# # Insurance Claim Prediction Using Linear Regression

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Gowtham\Finger tips\All Projects\Python + ML\ML Project - Linear Regression Insurance Prediction\insurance_cost_prediction_data.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df['region'].unique()


# In[8]:


df['smoker'].unique()


# In[9]:


df['sex'].unique()


# In[10]:


#encoding data
df=pd.get_dummies(df,drop_first=True)


# In[11]:


df.head()


# In[12]:


#scaling data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_ss=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
df_ss.head()


# In[13]:


X=df_ss.drop("charges",axis=1)
y=df_ss["charges"]


# In[14]:


#random_state Controls the shuffling applied to the data before applying the split.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)


# In[15]:


X_train.head()


# In[16]:


from sklearn.linear_model import LinearRegression
mod=LinearRegression()
mod.fit(X_train,y_train)


# In[17]:


y_pred=mod.predict(X_test)


# In[18]:


sns.distplot((y_test-y_pred),bins=50);


# # Model Evulation

# In[19]:


from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error


# In[20]:


#mean squre error
print(mean_squared_error(y_test,y_pred))


# In[21]:


#mean absolute erro
print(mean_absolute_error(y_test,y_pred))


# In[22]:


#R2 score
print(r2_score(y_test,y_pred))

