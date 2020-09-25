# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:20:59 2020

@author: zbfypn
"""
#%% 
import pandas as pd
import numpy as np
import seaborn as sb
import math
import matplotlib.pyplot as plt
import sklearn as sk
import scipy as sp



#%%

from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import math


house_data = pd.read_csv("kc_house_data.csv")

house_data.head()

house_data.describe()


#%%

house_data = house_data.sample(frac = 1).reset_index(drop = True)


x_data = house_data['sqft_living']

y_data = house_data['price']


#%%

x_data = np.array(x_data)
y_data = np.array(y_data)

x_data.shape

x_data = x_data.reshape(-1,1)
y_data = y_data.reshape(-1,1)





#%%
!pip install keras --upgrade
!pip install tensorFlow --upgrade

# pip install --user pkg-name

import warnings

warnings.filterwarnings('ignore')


#%%

from keras.models import Sequential
from keras.layers import Dense


ss = preprocessing.StandardScaler()
x_data = ss.fit_transform(x_data)



#%%

train_data = x_data[:-4322]

test_data = x_data[-4322:]




#%%

print("Training set: {}".format(train_data.shape))
print("Training set: {}".format(test_data.shape))


#%% Model building

train_labels = y_data[:-4322]
test_labels = y_data[-4322:]

# reordering the train data

order = np.argsort(np.random.random(train_labels.shape))


train_data = train_data[order]
train_labels = train_labels[order]


regr = linear_model.LinearRegression()


regr.fit(train_data, train_labels)

LinearRegression(copy_X = True, fit_intercept = True, n_jobs = 1, normalize = False)


train_predictions = regr.predict(train_data)
test_predictions = regr.predict(test_data)



#%% Model summary

print('Co-effcients : \n', regr.coef_)      # Get the slope
print('Bias : \n', regr.intercept_)         # Get the intercept


mse = mean_squared_error(test_labels, test_predictions)

print("MSE: %.2f" % mse)

math.sqrt(mse)

# R-square value for train data

print("r^2 score for training data: %.2f" % r2_score(train_labels, train_predictions))

print("r^2 score for testing data: %.2f" % r2_score(test_labels, test_predictions))


#%%

plt.figure(figsize = (8,8))

plt.scatter(train_data, train_labels, color = 'gray')

plt.plot(train_data, train_predictions, color = 'blue')

plt.show()


# For test data

plt.scatter(test_labels, test_predictions, color = 'gray')

plt.plot(train_data, train_predictions, color = 'blue')

plt.show()




#%% Model building using sequential keras model


model = Sequential()

model.add(Dense(1,
                input_dim = 1,
                activation = 'linear'
                )
          )


model.summary()

weights = model.layers[0].get_weights()


print(weights)

weights_initial = weights[0]
bias_initial = weights[1]


print("weigth : %.2f, b : %.2f" %(weights_initial, bias_initial))


#%% Compile the neural network

model.compile(optimizer = 'sgd',
              loss = 'mse',
              metrics = ['mae'])


history = model.fit(train_data,
                    train_labels,
                    validation_split = 0.2,
                    epochs = 300,
                    verbose = 0)


def plot_history(history):
    
    plt.figure()
    plt.xlabel('Epochs') 
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mae']),
             label = 'Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mae']),
             label = 'Val loss')
    
    plt.legend()
    plt.show()
    
    

plot_history(history)

weights = model.layers[0].get_weights

print(weights)


result = model.predict(train_data)

plt.scatter(train_data, train_labels,
            label = 'data', color = 'grey')
plt.plot(train_data, result, label = 'prediction')
plt.legend()
plt.show()


#%% Evaluating a Neural Network


[loss, mae] = model.evaluate(test_data,
                             test_labels,
                             verbose = 0)

math.sqrt(loss)


test_prediction = model.predict(test_data).flatten()


#%% plot test prediction

plt.figure(figsize = (8,8))

plt.scatter(test_labels.flatten(), test_predictions)

plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.show()

r2_score(train_labels, result)
r2_score(test_labels, test_predictions)


#%% Error plot

error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.show()



#%%





