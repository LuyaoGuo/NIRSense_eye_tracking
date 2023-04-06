# This file aims to signal preprocessing, signal processing and curve fitting the data obtained from the
# 'gaze_animation_full_term.py' in order to predict the eye traces
# Make sure gaze_animation_full_term.py is in the same folder

# Import
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
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import vg
from sklearn.metrics import explained_variance_score, r2_score

################################
##### Run Calibration Stage ####
################################

########################
# Signal Preprocessing #
########################
starting_time = datetime.datetime.now()
# find sensor and dot file for training
# make sure only one pair of sensor and dot data in this folder
sensor_file = glob.glob('sensor*')
dot_file = glob.glob('dot*')
remove_content = ["'", "[", "]"]
sensor_file = repr(sensor_file)
dot_file = repr(dot_file)
for content in remove_content:
    sensor_file = sensor_file.replace(content, '')
    dot_file = dot_file.replace(content, '')


df_dot = pd.read_csv(dot_file)
df_sensor = pd.read_csv(sensor_file)
dot_time = df_dot['timestamp']
sensor_time = df_sensor['timestamp']
dot_timestamp = []
for time in dot_time:
    time = time.split(' ')
    timestamp = time[1]
    timestamp = timestamp.split(':')
    hour = timestamp[0]
    minute = timestamp[1]
    second = timestamp[2]
    microsecond = timestamp[3]
    dot_timestamp.append(float(hour) + float(minute)/60 + float(second) / 3600 + float(microsecond) / (6e7*60))
    #dot_timestamp.append(float(minute) + float(second) / 60 + float(microsecond) / (6e7))

df_dot['timestamp_converted'] = dot_timestamp
original_dot_timestamp = dot_timestamp
sensor_timestamp=[]
for time in sensor_time:
    try: # remove first and last row of sensor file
        time = time.split(' ')
        timestamp = time[1]
        timestamp = timestamp.split(':')
        hour = timestamp[0]
        minute = timestamp[1]
        second = timestamp[2]
        microsecond = timestamp[3]
        sensor_timestamp.append(float(hour) + float(minute)/60 + float(second) / 3600 + float(microsecond) / (6e7*60))
        #sensor_timestamp.append(float(minute) + float(second) / 60 + float(microsecond) / (6e7))
    except: pass

df_sensor['timestamp_converted']=sensor_timestamp

# segment data
dot_timestamp = df_dot['timestamp_converted'].to_numpy()
sensor_timestamp = df_sensor['timestamp_converted'].to_numpy()

num_of_segment = int((len(dot_timestamp)-1)/9)
dot_timestamp = np.split(dot_timestamp[:-1], num_of_segment)

min_dot_time_per_segment = []
max_dot_time_per_segment = []
for i in range(len(dot_timestamp)):
    min_dot_time_per_segment.append(min(dot_timestamp[i]))
    max_dot_time_per_segment.append(max(dot_timestamp[i]))

# find index in sensor data within min and max of dot timestamps
sensor_segment_index = []
for i in range(len(min_dot_time_per_segment)):
    sensor_segment_index.append(np.where((sensor_timestamp > min_dot_time_per_segment[i]) & (sensor_timestamp < max_dot_time_per_segment[i]))[0][-1])
sensor_segment_index = [0]+sensor_segment_index

first_dot_second = original_dot_timestamp[0]
first_dot_list = []
first_sensor_list = []
for j in range(sensor_segment_index[0],sensor_segment_index[0+1]+1):
    if sensor_timestamp[j]<first_dot_second:
        first_sensor_list.append(sensor_timestamp[j])
        first_dot_list.append(first_dot_second)

dot_list = []
sensor_list = []
for i in range(15):
    for j in range(sensor_segment_index[i],sensor_segment_index[i+1]+1):
        for k in range(int(np.where((original_dot_timestamp==min_dot_time_per_segment[i]))[0]), \
                       int(np.where((original_dot_timestamp==max_dot_time_per_segment[i]))[0])+1):

            if original_dot_timestamp[k-1]<sensor_timestamp[j]<original_dot_timestamp[k]:
                dot_list.append(original_dot_timestamp[k])
                sensor_list.append(sensor_timestamp[j])
dot_list = np.concatenate([first_dot_list,dot_list])
sensor_list = np.concatenate([first_sensor_list, sensor_list])

test_list = pd.DataFrame({'expand_dot_timestamp': dot_list, 'expand_sensor_timestamp':sensor_list})
test_list.to_csv('combine_time.csv',index=False)


pd_combine_time = pd.read_csv('combine_time.csv')
df_time_with_sensor = pd.concat([pd_combine_time,df_sensor],axis=1)
df_time_with_sensor.to_csv('preprocessed_data_test.csv',index=False)
df1 = df_time_with_sensor
df_dot.to_csv('test_dot.csv')
df_dot_2 = pd.read_csv('test_dot.csv')
df_combined = pd.read_csv('preprocessed_data_test.csv')
df_combined = df_combined.set_index('expand_dot_timestamp')
df_dot_2 = df_dot_2.set_index('timestamp_converted')
df_all = df_combined.join(df_dot_2, lsuffix='_left', rsuffix='_right')
df_all = df_all[['dot_x','dot_y','LL','LC','LR','RL','RC','RR']]
df_all.to_csv('preprocessed_data.csv',index=False)

# remove some files used during calibration
os.remove('preprocessed_data_test.csv')
os.remove('combine_time.csv')
os.remove('test_dot.csv')
finishing_time = datetime.datetime.now()
consumed_time = finishing_time-starting_time
print('Total time consumed: {}'.format(consumed_time))


################################
######### Extract data #########
################################
# This section aims to extract all the data satisfying the '20 degree criteria'
# File 'preprocessed_for_accu.csv' will be generated
# which will be used to calculate the accuracy result
# This code is run under the assumption that the participant's eyes are facing to the middle point the screen

# Read data from the preprocessed file
df_all = pd.read_csv('preprocessed_data.csv')
sensors = ['LL', 'LC', 'LR', 'RL', 'RC', 'RR']

# dot data
dot_x = df_all['dot_x'].to_numpy()
dot_y = df_all['dot_y'].to_numpy()


# Apply filter to data for different points separately and then concatenate together
# Function find index of data where the points are different
# Find by the position of baseline data
# ls1 is the list of number that baseline data in x axis

def find_index_for_each_point(ls1):
    index_of_point = [0]
    ls1 = ls1.tolist()
    for i in range(len(ls1)-1):
        if ls1[i+1] != ls1[i]:
            index_of_point.append(i+1)
    return index_of_point

index_of_each_point = find_index_for_each_point(dot_x)
index_of_each_point = index_of_each_point + [len(dot_x)]
sensors_raw = df_all[sensors]
sensors_filtered = []
for i in range(len(index_of_each_point)-1):
    sensors_filtered.append(\
    (sensors_raw.iloc[int(index_of_each_point[i]):int(index_of_each_point[i+1]),:].apply(savgol_filter, window_length=31, polyorder=2)))
sensors_filtered = pd.concat(sensors_filtered)
df_all[['LL_filtered', 'LC_filtered', 'LR_filtered', 'RL_filtered', 'RC_filtered', 'RR_filtered']] = sensors_filtered
print(df_all.head())
loop_index_temp = df_all.index[(df_all['dot_x'] == 0) & (df_all['dot_y'] == 8)].tolist()
loop_index = []
for i in range(len(loop_index_temp)-1):
    if loop_index_temp[i+1]-loop_index_temp[i]>1:
        loop_index.append(loop_index_temp[i+1])
loop_index = [0]+loop_index

accu_degree_dot_index_1 = []
accu_degree_dot_index_2 = []
accu_degree_dot_index_3 = []
accu_degree_dot_index_4 = []
accu_degree_dot_index_5 = []
for i in range(len(dot_x)):
    if dot_x[i]==4 and dot_y[i]==8 and i>loop_index[2]:
        accu_degree_dot_index_1.append(i)
    if (dot_x[i]==3 or dot_x[i]==4 or dot_x[i]==5) \
            and dot_y[i]==6 and i>loop_index[2]:
        accu_degree_dot_index_2.append(i)
    if (dot_x[i]==2 or dot_x[i]==3 or dot_x[i]==4 or dot_x[i]==5 or dot_x[i]==6) \
            and dot_y[i]==4 and i>loop_index[2]:
        accu_degree_dot_index_3.append(i)
    if (dot_x[i]==3 or dot_x[i]==4 or dot_x[i]==5) \
            and dot_y[i]==2 and i>loop_index[2]:
        accu_degree_dot_index_4.append(i)
    if dot_x[i]==4 and dot_y[i]==0 and i>loop_index[2]:
        accu_degree_dot_index_5.append(i)

accu_degree_dot_index = np.concatenate([accu_degree_dot_index_1,accu_degree_dot_index_2,accu_degree_dot_index_3,\
                                        accu_degree_dot_index_4,accu_degree_dot_index_5])

df_accu = df_all.loc[accu_degree_dot_index]
df_accu.to_csv('preprocessed_for_accu.csv',index=False)
df_all.to_csv('preprocessed_data_filtered.csv',index=False)
######################
# Run Training Stage #
######################

# collect data from csv

df_all = pd.read_csv('preprocessed_data_filtered.csv')
df_test_for_accu_degree = pd.read_csv('preprocessed_for_accu.csv')

# Use filtered sensor data
sensors_filtered = ['LL_filtered', 'LC_filtered', 'LR_filtered', 'RL_filtered', 'RC_filtered', 'RR_filtered']

# dot data
dot_x = df_all['dot_x']
dot_y = df_all['dot_y']

# find start index of each loop
temp_index = []
for i in range(len(dot_x)):
    if dot_x[i]==0 and dot_y[i]==8:
        temp_index.append(i)
start_index = []
for j in range(len(temp_index)-1):
    if temp_index[j+1]-temp_index[j] > 1:
        start_index.append(temp_index[j+1])
start_index = [0]+start_index

# Train data get from the df_all dataframe
X_train = df_all.loc[start_index[0]:start_index[2], sensors_filtered].values
dot_x_train = df_all.loc[start_index[0]:start_index[2], ['dot_x']].values
dot_y_train = df_all.loc[start_index[0]:start_index[2], ['dot_y']].values

# Test data get from the df_test_for_accu_degree dataframe
X_test = df_test_for_accu_degree.loc[:, sensors_filtered].values
dot_x_test = df_test_for_accu_degree.loc[:, ['dot_x']].values
dot_y_test = df_test_for_accu_degree.loc[:, ['dot_y']].values

# Normalise sensors
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler2 = MinMaxScaler()
scaler2.fit(X_test)
X_test = scaler.transform(X_test)

# Smooth/filter sensors
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

from sklearn.preprocessing import PolynomialFeatures
# for creating pipeline
from sklearn.pipeline import Pipeline
# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=2)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(X_train,dot_x_train)
poly_pred=pipe.predict(X_test)
pipe2=Pipeline(Input)
pipe2.fit(X_train,dot_y_train)
poly_pred2=pipe.predict(X_test)
plt.scatter(poly_pred,poly_pred2)
plt.scatter(dot_x_test,dot_y_test)
plt.show()

#accuracy evaluation
accuracy_dot_x = explained_variance_score(dot_x_test, poly_pred)
accuracy_dot_y = explained_variance_score(dot_y_test, poly_pred2)
accuracy_percent = (accuracy_dot_x+accuracy_dot_y)/2
print('The accuracy in percentage is {:0.2f}%'.format(accuracy_percent*100))

#calculate accuracy in terms of degree
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

# transform into 3d vectors
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
distance_between_real_dot_and_centre = []
distance_between_predict_dot_and_centre = []

for i in range(len(dot_x_test)):
    vec_real_dots[i,0]=dot_x_test[i]*times_screen_width-154/2
    vec_real_dots[i,1]=dot_y_test[i]*times_screen_height-87/2
    vec_real_dots[i,2] = 120

    vec_predict_dots[i, 0] = poly_pred[i]*times_screen_width-154/2
    vec_predict_dots[i, 1] = poly_pred2[i]*times_screen_height-87/2
    vec_predict_dots[i, 2] = 120
    angle_difference.append((angle(vec_real_dots[i,:],vec_predict_dots[i,:])))

# calculate mean value in angle_difference
accuracy_degree = sum(angle_difference)/len(angle_difference)
print('The accuracy in degree is {:0.2f} degree'.format(accuracy_degree))
