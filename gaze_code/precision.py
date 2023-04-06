# This
#
import glob
import re
from subprocess import call
import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import math
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

######################
### Precision Stage ##
######################

#collect data from csv
df_all = pd.read_csv('preprocessed_data_filtered.csv')
df_test_for_accu_degree = pd.read_csv('preprocessed_for_accu.csv')

sensors_filtered = ['LL_filtered', 'LC_filtered', 'LR_filtered', 'RL_filtered', 'RC_filtered', 'RR_filtered']

#dot data
dot_x = df_all['dot_x']
dot_y = df_all['dot_y']

#find start index of each loop
temp_index = []
for i in range(len(dot_x)):
    if dot_x[i]==0 and dot_y[i]==8:
        temp_index.append(i)
start_index = []
for j in range(len(temp_index)-1):
    if temp_index[j+1]-temp_index[j] > 1:
        start_index.append(temp_index[j+1])
start_index = [0]+start_index
# Train data
X_train = df_all.loc[start_index[0]:start_index[2], sensors_filtered].values
dot_x_train = df_all.loc[start_index[0]:start_index[2], ['dot_x']].values
dot_y_train = df_all.loc[start_index[0]:start_index[2], ['dot_y']].values
# Test data
X_test = df_test_for_accu_degree.loc[:, sensors_filtered].values
dot_x_test = df_test_for_accu_degree.loc[:, ['dot_x']].values
dot_y_test = df_test_for_accu_degree.loc[:, ['dot_y']].values

#Normalise sensors
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler2 = MinMaxScaler()
scaler2.fit(X_test)
X_test = scaler.transform(X_test)

# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(X_train,dot_x_train)
poly_pred=pipe.predict(X_test)
pipe2=Pipeline(Input)
pipe2.fit(X_train,dot_y_train)
poly_pred2=pipe.predict(X_test)

from sklearn.metrics import explained_variance_score, r2_score
#accuracy evaluation
accuracy_dot_x = explained_variance_score(dot_x_test, poly_pred)
accuracy_dot_y = explained_variance_score(dot_y_test, poly_pred2)
accuracy_percent = (accuracy_dot_x+accuracy_dot_y)/2
# print(accuracy_percent)

#calculate accuracy in terms of degree
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

import vg

vec_real_dots = np.zeros((len(dot_x_test),3))
vec_human = np.zeros((len(dot_x_test),3))
vec_predict_dots = np.zeros((len(dot_x_test),3))
angle_between_real_dot_and_human = []
angle_between_predict_dot_and_human = []
angle_difference = []

times_screen_width = 154/8 #number of value to times up to convert (x,y) to cm in real screen
times_screen_height = 87/8
distance_between_real_dot_and_human = []
distance_between_predict_dot_and_human = []
angle_predict_to_human=[]

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2): # return in radians
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# index of each dots
index_of_each_dot = [0]
for i in range(len(dot_x_test)-1):
    if dot_x_test[i+1]-dot_x_test[i]>0:
        index_of_each_dot.append(i+1)
index_of_each_dot = index_of_each_dot + [len(dot_x_test)]

for j in range(len(index_of_each_dot)-1):
    for i in range(index_of_each_dot[j],index_of_each_dot[j+1]):
        vec_predict_dots[i, 0] = poly_pred[i] * times_screen_width - 154 / 2
        vec_predict_dots[i, 1] = poly_pred2[i] * times_screen_height - 87 / 2
        vec_predict_dots[i, 2] =120
        vec_human[i, 0] = 0
        vec_human[i, 1] = 0
        vec_human[i, 2] = 0

for i in range(1,len(dot_x_test)):
    angle_predict_to_human.append(math.degrees(angle_between(vec_predict_dots[i-1,:],vec_predict_dots[i,:])))

def rmsValue(arr, n):
    square = 0
    mean = 0.0
    root = 0.0
    # Calculate square
    for i in range(0, n):
        square += (arr[i] ** 2)
    # Calculate Mean
    mean = (square / (float)(n))
    # Calculate Root
    root = math.sqrt(mean)
    return root
precision_rms = rmsValue(angle_predict_to_human,len(angle_predict_to_human))
print('The precision in degree is {:0.2f} degree'.format(precision_rms))
