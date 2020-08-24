#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd

#loading wine dataset from sklearn
from sklearn.datasets import load_wine
wine=load_wine()
dir(wine)


# In[18]:


#Data Frame is created from Wine Dataset
df=pd.DataFrame(wine.data,columns=wine.feature_names)
df.head()


# In[19]:


#target column
target=wine.target


# In[20]:


#splitting test and train datasets
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test=train_test_split(df,target,test_size=0.2)


# In[21]:


#model is for Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()


# In[22]:


#model1 is for Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()


# In[23]:


#Multinomial model is trained
model.fit(X_train,y_train)


# In[24]:


#Gaussian Model is trained
model1.fit(X_train,y_train)


# In[25]:


#Score of Multinomial model
model.score(X_test,y_test)


# In[26]:


#score of Gaussian model
model1.score(X_test,y_test)


# In[27]:


#Predicting Values using Multinomial model
model.predict(X_test[:10])


# In[28]:


#Values from the dataset
y_test[:10]


# In[29]:


#Storing all the predicted values for X_test in y_predicted using Multinomial model
y_predicted=model.predict(X_test)

#Generating Confusion matrix for Mutinomial model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)


# In[30]:


#Plotting confusion matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[31]:


#CONCLUSION: Gaussian Model is Better than Multinomial model in this dataset. 
#We dont need to plot confusion matrix for Gaussian model as the score is 1 so all the predictions are correct.

