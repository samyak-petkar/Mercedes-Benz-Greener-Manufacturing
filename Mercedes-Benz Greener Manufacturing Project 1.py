#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[75]:


train = pd.read_csv('MERCtrain.csv')
test  = pd.read_csv('MERCtest.csv')


# In[76]:


train.head()


# In[77]:


train.info()


# In[78]:


train.y.value_counts()


# In[79]:


y_train = train['y'].values
y_train


# In[80]:


col_x =[c for c in train.columns if 'X' in c]
print(len(col_x))


# In[81]:


print(train[col_x].dtypes.value_counts())


# In[86]:


final_col = list(set(train.columns) - set(['ID','y']))


# In[88]:


x_train = train[final_col]
# x_train
x_test = test[final_col]
# x_test


# In[89]:


def detect(df):
    if df.isnull().any().any():
        print("Yes")
    else:
        print("No")

detect(x_train)
detect(x_test)


# In[90]:


for column in final_col:
    check = len(np.unique(x_train[column]))
    if check == 1:
        x_train.drop(column, axis = 1) 
        x_test.drop(column, axis = 1)
    if check > 2: # Column is categorical; hence mapping to ordinal measure of value
        mapit = lambda x: sum([ord(digit) for digit in x])
        x_train[column] = x_train[column].apply(mapit)
        x_test[column] = x_test[column].apply(mapit)

x_train.head()


# In[91]:


from sklearn.decomposition import PCA
n_comp = 12
pca = PCA(n_components = n_comp, random_state = 42)
pca_result_train = pca.fit_transform(x_train)
pca_result_test = pca.transform(x_test)


# In[92]:


#XGboost
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# In[93]:


x_train, x_valid, y_train, y_valid = train_test_split(pca_result_train, y_train, test_size = 0.2, random_state = 42)


# In[94]:


f_train = xgb.DMatrix(x_train, label = y_train)
f_valid = xgb.DMatrix(x_valid, label = y_valid)
f_test = xgb.DMatrix(x_test)
f_test = xgb.DMatrix(pca_result_test)


# In[95]:


params = {}
params['objective'] = 'reg:linear'
params['eta'] = 0.02
params['max_depth'] = 4


# In[96]:


def scorer(m, w):
    labels = w.get_label()
    return 'r2', r2_score(labels, m)

final_set = [(f_train, 'train'), (f_valid, 'valid')]

pred = xgb.train(params, f_train, 1000, final_set, early_stopping_rounds=50, feval=scorer, maximize=True, verbose_eval=10)


# In[97]:


p_test = pred.predict(f_test)
p_test


# In[98]:


Predicted_Data = pd.DataFrame()
Predicted_Data['y'] = p_test
Predicted_Data.head()


# In[ ]:




