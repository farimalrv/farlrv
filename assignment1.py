#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set()
A=pd.read_csv('dataA (1).csv',header=None,delimiter=',')
A


# In[2]:


#dataA
A1=np.genfromtxt("dataA (1).csv",delimiter=',',skip_header=0)
A1


# In[3]:


#because np.cov() works through rows as variables
A1T=np.transpose(A1)
A1T


# In[4]:


#covariance matrix of data A
Cov=np.cov(A1T)
Cov


# In[5]:


#mean of each column of the data A
Mean=np.mean(A1,axis=0)
Mean


# In[6]:


B=pd.read_csv('dataB (1).csv',header=None,delimiter=',')
B


# In[7]:


#data B
B1=np.genfromtxt("dataB (1).csv",delimiter=',',skip_header=0)
B1


# In[8]:


#because np.cov() works through rows as variables
B1T=np.transpose(B1)
B1T


# In[9]:


#covariance matrix of data B
Cov=np.cov(B1T)
Cov


# In[10]:


#mean of each column of the data B
Mean=np.mean(B1,axis=0)
Mean


# In[ ]:


#answer of q3
#the answers could be different because of preproccesing of the data,here we didn't preprocessed or scaled data.

