#!/usr/bin/env python
# coding: utf-8

# # Importing the required Libraries

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Loading dataset

# In[3]:


data = pd.read_csv("C:/Users/MTN-SIMREG/PycharmProjects/Diabetes_prediction/DiabetesPrediction/my_site/Data/diabetes.csv")


# In[4]:


data.head(20)


# In[5]:


data.tail(10)


# # Exploratory data analysis

# In[7]:


data.describe()


# In[8]:


data.info


# In[14]:


sn.pairplot(data)


# # Corelation Matrix

# In[15]:


correlation = data.corr()
correlation


# In[19]:


sn.heatmap(correlation)
plt.show()


# # Model Training

# In[28]:


x = data.drop('Outcome', axis = 1)
y = data['Outcome']

X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, Y_train)


# # Making Predictions

# In[30]:


predictions = model.predict(X_test)
predictions


# In[33]:


accuracy = accuracy_score(predictions, Y_test)
accuracy


# In[ ]:




