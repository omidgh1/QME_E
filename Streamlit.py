#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import numpy as np
import sklearn
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[2]:


st.title('QME Course')
st.title('Group E')


# In[3]:


df = pd.read_csv('numerical_info_final.csv')
del df['Unnamed: 0']
del df['Unnamed: 0.1']
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df.select_dtypes(include=numerics)
cols = newdf.columns.to_list()
newdf


# In[4]:


def reg(data,target,col_list):  
    r2 = []
    for i in col_list:
        data.dropna(subset = [i], inplace=True)
        data.dropna(subset = [target], inplace=True)
        x = data[[i]]
        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
        LR = LinearRegression()
        # fitting the training data
        LR.fit(x_train,y_train)
        y_prediction =  LR.predict(x_test)
        # predicting the accuracy score
        score=r2_score(y_test,y_prediction)
        r2.append(score)
    
    x = data[col_list]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    LR = LinearRegression()
    # fitting the training data
    LR.fit(x_train,y_train)
    y_prediction =  LR.predict(x_test)
    # predicting the accuracy score
    score=r2_score(y_test,y_prediction)
    
    intercept = LR.intercept_
    features = pd.DataFrame(LR.coef_, x.columns, columns=['coefficient'])
    features.coefficient = features.coefficient.abs()
    stdevs = []
    for i in x.columns:
        stdev = data[i].std()
        stdevs.append(stdev)
    features["stdev"] = stdevs
    features["importance"] = features["coefficient"] * features["stdev"]
    features['importance_normalized'] = (100*features['importance']) / features['importance'].max()
    features['R2_Score'] = r2
    return features


# In[5]:


Countries = st.selectbox('Countries',['All countries','Developed Countries','Developing Countries'])
if Countries == 'Developed Countries':
    data = newdf[newdf['Developed']==1]
elif Countries == 'Developing Countries':
    data = newdf[newdf['Developed']==1]
else:
    data = newdf


# In[6]:


target = st.selectbox('Target',cols,index = cols.index('Total Current revenues (EURO)'))


# In[7]:


independent_vars = st.multiselect('Select your variables', cols,['Erasmus total incoming students',
                                                                      'Erasmus total outgoing students'])


# In[10]:


regression = reg(data,target,independent_vars)
regression

