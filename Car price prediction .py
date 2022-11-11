#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[2]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/CarPrice.csv')


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.CarName.unique()


# In[8]:


sns.set_style('whitegrid')
plt.figure(figsize=(15,10))
sns.distplot(data.price)
plt.show()


# In[9]:


data.corr()


# In[10]:


correlations=data.corr()
plt.figure(figsize=(20,15))
sns.heatmap(correlations, annot=True)
plt.show()


# In[11]:


data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]


# In[12]:


x=np.array(data.drop(['price'],1))
y=np.array(data['price'])


# In[13]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)
model=DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

from sklearn.metrics import mean_absolute_error
model.score(xtest, predictions)

