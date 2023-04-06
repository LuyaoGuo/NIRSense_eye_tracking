# This code aims to train and test the result of pursuit experiment
# Results of accuracy and precision will be generated
# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

######################
# Run Training Stage #
######################


#collect data from csv
df_pursuit = pd.read_csv('pursuit_preprocessed.csv')

sensors = ['LL', 'LC', 'LR', 'RL', 'RC', 'RR']
sensors_filtered = ['LL_filtered', 'LC_filtered', 'LR_filtered', 'RL_filtered', 'RC_filtered', 'RR_filtered']


# Find index of each line
movement_number = df_pursuit.loc[:, ['movement_number']].values
def find_index_for_each_line(ls1):
    index_of_point = [0]
    ls1 = ls1.tolist()
    for i in range(len(ls1)-1):
        if ls1[i+1] != ls1[i]:
            index_of_point.append(i+1)
    return index_of_point
index_of_each_line = find_index_for_each_line(movement_number) + [len(movement_number)]

# Apply filter to each line separately
sensors_raw = df_pursuit[sensors]
sensors_filtered = []
for i in range(len(index_of_each_line)-1):
    sensors_filtered.append(\
    (sensors_raw.iloc[int(index_of_each_line[i]):int(index_of_each_line[i+1]),:].apply(savgol_filter, window_length=31, polyorder=2)))
sensors_filtered = pd.concat(sensors_filtered)
df_pursuit[['LL_filtered', 'LC_filtered', 'LR_filtered', 'RL_filtered', 'RC_filtered', 'RR_filtered']] = sensors_filtered

# Seprate the training data and testing data from database
# movement number 2 3 4 5 are training data
# movement number 0 1 are testing data
X_train = []
dot_x_train = []
dot_y_train = []

X_test = []
dot_x_test = []
dot_y_test = []

for i in range(4):
    # train
    X_train.append(df_pursuit.loc[index_of_each_line[6*i+2]:index_of_each_line[6*i+6], 'LL_filtered':'RR_filtered'])
    dot_x_train.append(df_pursuit.loc[index_of_each_line[6*i+2]:index_of_each_line[6*i+6], ['dot_x']])
    dot_y_train.append(df_pursuit.loc[index_of_each_line[6*i+2]:index_of_each_line[6*i+6],['dot_y']])
    # test
    X_test.append(df_pursuit.loc[index_of_each_line[6 * i]:index_of_each_line[6 * i + 2], 'LL_filtered':'RR_filtered'])
    dot_x_test.append(df_pursuit.loc[index_of_each_line[6 * i]:index_of_each_line[6 * i + 2], ['dot_x']])
    dot_y_test.append(df_pursuit.loc[index_of_each_line[6 * i]:index_of_each_line[6 * i + 2], ['dot_y']])

# concat
X_train = pd.concat(X_train)
dot_x_train = pd.concat(dot_x_train)
dot_y_train = pd.concat(dot_y_train)

X_test = pd.concat(X_test)
dot_x_test = pd.concat(dot_x_test)
dot_y_test = pd.concat(dot_y_test)

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
#poly_pred = poly_pred_x2.reshape(len(poly_pred_x2),1)
pipe2=Pipeline(Input)
pipe2.fit(X_train,dot_y_train)
poly_pred2=pipe.predict(X_test)

# plot
plt.rcParams['axes.facecolor'] = 'black'
plt.scatter(dot_x_test,dot_y_test,s=5,color='gray',label='Baseline Lines')
plt.scatter(poly_pred,poly_pred2,s=2,color='white',label='Pursuit traces')
plt.legend(facecolor='white',fontsize=16,loc=1)
plt.xticks([])
plt.yticks([])
plt.xlabel('Screen in x axis',Fontsize=20)
plt.ylabel('Screen in y axis',Fontsize=20)
plt.show()


from sklearn.metrics import explained_variance_score, r2_score
#accuracy evaluation
accuracy_dot_x = explained_variance_score(dot_x_test, poly_pred)
accuracy_dot_y = explained_variance_score(dot_y_test, poly_pred2)
accuracy_percent = (accuracy_dot_x+accuracy_dot_y)/2
print('The accuracy in percentage is {:0.2f}%'.format(accuracy_percent*100))

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
for i in range(len(dot_x_test)):
    vec_predict_dots[i, 0] = poly_pred[i] * times_screen_width - 154 / 2
    vec_predict_dots[i, 1] = poly_pred2[i] * times_screen_height - 87 / 2
    vec_predict_dots[i, 2] = 120
    vec_human[i, :] = [0,0,0]
    angle_predict_to_human.append(np.arctan(abs(poly_pred[i] * times_screen_width - 154 / 2)/abs(poly_pred2[i] * times_screen_height - 87 / 2)))


precision = sum(angle_predict_to_human)/len(angle_predict_to_human)
print('The precision in degree is {:0.2f} degree'.format,precision)

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


distance_between_real_dot_and_human = []
distance_between_predict_dot_and_human = []
distance_between_real_dot_and_centre = []
distance_between_predict_dot_and_centre = []

dot_x_test = dot_x_test.to_numpy()
dot_y_test = dot_y_test.to_numpy()
for i in range(len(dot_x_test)):
    vec_real_dots[i,0]=dot_x_test[i]*times_screen_width-154/2
    vec_real_dots[i, 1] = dot_y_test[i]*times_screen_height-87/2
    vec_real_dots[i, 2] = 120

    vec_predict_dots[i, 0] = poly_pred[i]*times_screen_width-154/2
    vec_predict_dots[i, 1] = poly_pred2[i]*times_screen_height-87/2
    vec_predict_dots[i, 2] = 120
    angle_difference.append((angle(vec_real_dots[i,:],vec_predict_dots[i,:])))

# calculate mean value in angle_difference
accuracy_degree = sum(angle_difference)/len(angle_difference)
print('The accuracy in degree is {:0.2f} degree'.format(accuracy_degree))
