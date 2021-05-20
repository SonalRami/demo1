#!/usr/bin/env python
# coding: utf-8

# ## How to Fit Regression Data with CNN Model in Python

# #### Necessary libraries
# * Numpy
# * Pandas
# * Scikit.learn
# * Tensorflow
# * Keras
# * Matplotlib
# 

# ### Table of Contents
# 
# * [Step1: Data Loading](#Data_loading) 
# * [Step2: Preparation of training and testing samples](#Prep_train_test_samples)
# * [Step3: Building Neural Network](#Building_NN)
# * [Step4: Fit model](#fit_model)
# * [Step5: Predictions](#predictions)
# * [Step6: Visualizing the results](#results)

# ### Step1: Data Loading <a class="anchor" id="Data_loading"></a>

# In[48]:


from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from pylab import rcParams


register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10


# In[49]:


#load dataset
boston = load_boston()


# In[50]:


x, y = boston.data, boston.target
print(x.shape) 


# In[51]:


#reshape data for Conv1D model
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)


# ### Step2: Preparation of training and testing samples <a class="anchor" id="Prep_train_test_samples"></a>

# In[52]:


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.20) 


# In[53]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ### Step3: Building Neural Network <a class="anchor" id="Building_NN"></a>

# In[54]:


model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(13, 1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
 
model.summary()


# ### Step4: Fit model <a class="anchor" id="fit_model"></a>

# In[57]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=12,epochs=200, verbose=1)


# ### Step5: Predictions <a class="anchor" id="predictions"></a>

# In[58]:


y_pred = model.predict(X_test)


# In[59]:


print(model.evaluate(X_train, y_train))


# In[60]:


print("MSE: %.4f" % mean_squared_error(y_test, y_pred))


# ### Step6: Visualizing the results <a class="anchor" id="results"></a>

# In[61]:


x_ax = range(len(y_pred))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="green", label="predicted")
plt.legend()
plt.show()


# ### two things remaining
# 1. Prevent overfitting
# 2. Tune Hyperparameters

# In[ ]:




