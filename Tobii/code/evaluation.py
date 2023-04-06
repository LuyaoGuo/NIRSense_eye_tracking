import csv
import numpy as np
import pandas as pd
f_record = open('recording_tobii_0_8_8_8_2022_12_13_18_12_27.csv')
line = f_record.readlines()[1:]
f_record.close()
lines = []
[lines.append(line) for line in line]
record_time = [i.split(',\n')[0] for i in lines]
###############for this test only, accendentally set time to utctime##############
#### covert hour 02 to 13
record_hour = [float(x)+11 for x in [j.split(':')[0] for j in [i.split(' ')[-1] for i in record_time]]]
record_minute = [float(x) for x in [j.split(':')[1] for j in [i.split(' ')[-1] for i in record_time]]]
record_second = [float(x) for x in [j.split(':')[2] for j in [i.split(' ')[-1] for i in record_time]]]
record_microsecond = [float(x) for x in [j.split(':')[3] for j in [i.split(' ')[-1] for i in record_time]]]
record_timestamps = []
record_timestamps.append([record_minute[i]*60+record_second[i]+record_microsecond[i]/(6e7/60) for i in range(len(record_hour))])
record_timestamp = []
for i in range(len(record_timestamps[0])-1):
    if float(record_timestamps[0][i+1])-float(record_timestamps[0][i]) > 0:
        record_timestamp.append(record_timestamps[0][i+1])

df_record = pd.DataFrame(list(zip(record_timestamp)), columns=['record_timestamp'])

# Baseline timestamp
f2 = open('dot_data_0_8_8_8_2022_12_13_18_12_27.csv')
dot_data = f2.readlines()[1:]
f2.close()

lines2 = []
[lines2.append(line) for line in dot_data]
dot_times = [j.split(' ')[-1] for j in [i.split(',')[0] for i in lines2]]
dot_timestamp = []
for time in dot_times:
    timestamp = time.split(':')
    hour = timestamp[0]
    minute = timestamp[1]
    second = timestamp[2]
    microsecond = timestamp[3]
    # dot_timestamp.append(float(hour) + float(minute) / 60 + float(second) / 3600 + float(microsecond) / (6e7 * 60))
    dot_timestamp.append(float(minute)*60+float(second)+ float(microsecond) / (6e7/60))

#create pd dataframe for baseline
# df_baseline=pd.DataFrame(dot_time_diff,columns=['baseline_timestamps'])
df_baseline = pd.DataFrame(list(zip(dot_timestamp, [i.split(',')[1] for i in lines2])), \
                           columns=['baseline_timestamp','baseline_x'])
# df_baseline['baseline_x'] = pd.DataFrame(list(zip([i.split(',')[1] for i in lines2])))
df_baseline['baseline_y'] = pd.DataFrame(list(zip([j.split("\n")[0] for j in [i.split(',')[2] for i in lines2]])))
df_baseline['baseline_x_cm'] = pd.read_csv('cm.csv').iloc[:]['dot_x_cm']
df_baseline['baseline_y_cm'] = pd.read_csv('cm.csv').iloc[:]['dot_y_cm']
#segment data
dot_timestamp = df_baseline['baseline_timestamp'].to_numpy()
record_timestamp = df_record['record_timestamp'].to_numpy()
original_dot_timestamp = df_baseline['baseline_timestamp'].tolist()
original_record_timestamp = df_record['record_timestamp'].tolist()

num_of_segment = int((len(dot_timestamp)-1)/9) #=15
dot_timestamp = np.split(dot_timestamp[:-1], num_of_segment) #the last timestamp is not used

min_dot_time_per_segment = []
max_dot_time_per_segment = []
for i in range(len(dot_timestamp)):
    min_dot_time_per_segment.append(min(dot_timestamp[i]))
    max_dot_time_per_segment.append(max(dot_timestamp[i]))

#find index in sensor data within min and max of dot timestamps
record_segment_index = []
for i in range(len(min_dot_time_per_segment)):
    record_segment_index.append(np.where((record_timestamp > min_dot_time_per_segment[i]) & (record_timestamp < max_dot_time_per_segment[i]))[0][-1])
record_segment_index = [0]+record_segment_index

# special case for the first second:
# the grouping has to be calculated seperately (eg.55.78737175)
first_dot_second = original_dot_timestamp[0]
first_dot_list = []
first_record_list = []
for j in range(record_segment_index[0],(record_segment_index[0+1]+1)):
    if record_timestamp[j]<first_dot_second:
        first_record_list.append(record_timestamp[j])
        first_dot_list.append(first_dot_second)

dot_list = []
record_list = []
for i in range(15):
    for j in range(record_segment_index[i],record_segment_index[i+1]+1):
        for k in range(int(np.where((original_dot_timestamp==min_dot_time_per_segment[i]))[0]), \
                       int(np.where((original_dot_timestamp==max_dot_time_per_segment[i]))[0])+1):
            if original_dot_timestamp[k-1]<record_timestamp[j]<original_dot_timestamp[k]:
                #test_list.append(sensor_timestamp[j],original_dot_timestamp[k+1])
                dot_list.append(original_dot_timestamp[k])
                record_list.append(record_timestamp[j])

dot_list = np.concatenate([first_dot_list,dot_list])
record_list = np.concatenate([first_record_list, record_list])
df_record_and_baseline_times = pd.DataFrame({'baseline_timestamp': dot_list,\
                                             'record_timestamp': record_list})
df_record_and_baseline_times = df_record_and_baseline_times.set_index('baseline_timestamp')
df_baseline = df_baseline.set_index('baseline_timestamp')
df_all = df_record_and_baseline_times.join(df_baseline, lsuffix='_left', rsuffix='_right')
df_all.to_csv('test.csv')

import re
import matplotlib.pyplot as plt
import pandas as pd
import csv

# check tobii folder name for starting time
tobii_start_time = '18:12:16' # The actual time is 18:12:09
tobii_start_time =float(tobii_start_time.split(':')[1])*60+\
                   float(tobii_start_time.split(':')[2])

f = open('gazedata')
info = f.readlines()
f.close()
lines = []
[lines.append(line) for line in info]
data_2d = [i.split(',"')[2] for i in lines]
data_2d_x = [j.split(',')[0] for j in [i.split(':[')[-1] for i in data_2d]]
data_2d_y = [z.split(']')[0] for z in [j.split(',')[-1] for j in [i.split(':[')[-1] for i in data_2d]]]
# print(type(data_2d_y))
tobii_timestamp_temp = [j.split('":')[1] for j in [i.split(',"')[1] for i in lines]]
# print(tobii_timestamp)
x_axis_2d = []
y_axis_2d = []
# get rid of unnormal rows
tobii_timestamp = []
for i in range(len(data_2d_x)):
    try:
        if isinstance(float(data_2d_x[i]), float):
            x_axis_2d.append(float(data_2d_x[i]))
            tobii_timestamp.append(float(tobii_timestamp_temp[i]))
        if isinstance(float(data_2d_y[i]), float):
            y_axis_2d.append(float(data_2d_y[i]))
    except: pass

# put tobii data to a pd.dataframe
df_tobii=pd.DataFrame(tobii_timestamp,columns=['tobii_timestamp'])
df_tobii['2d_x'] = pd.DataFrame(x_axis_2d)
df_tobii['2d_y'] = pd.DataFrame(y_axis_2d)

# get timestamp in dot data
# matching with tobii data
# dot timestamp - tobii data starting title (check folder name)
f2 = open('test.csv')
dot_data = f2.readlines()[1:]
f2.close()


lines2 = []
[lines2.append(line) for line in dot_data]
baseline_times = [i.split(',')[0] for i in lines2]
record_times = [i.split(',')[1] for i in lines2]
baseline_x = [float(i.split(',')[2]) for i in lines2]
baseline_y = [float(i.split(',')[3]) for i in lines2]
baseline_x_cm = [float(i.split(',')[4]) for i in lines2]
baseline_y_cm = [float(i.split(',')[5]) for i in lines2]


baseline_time_diff = []
record_time_diff = []
baseline_time_diff.append([(float(i)-tobii_start_time) for i in baseline_times])
record_time_diff.append([(float(i)-tobii_start_time) for i in record_times])
# print(dot_time_diff)
baseline_time_diff = [j for sub in baseline_time_diff for j in sub]
record_time_diff = [j for sub in record_time_diff for j in sub]

#create pd dataframe for baseline
# df_baseline=pd.DataFrame(dot_time_diff,columns=['baseline_timestamps'])
df_baseline = pd.DataFrame(list(zip(baseline_time_diff, record_time_diff)), \
                           columns=['baseline_timestamp_diff','record_timestamp_diff'])
df_baseline['baseline_x'] = pd.DataFrame(baseline_x)
df_baseline['baseline_y'] = pd.DataFrame(baseline_y)
df_baseline['baseline_x_cm'] = pd.DataFrame(baseline_x_cm)
df_baseline['baseline_y_cm'] = pd.DataFrame(baseline_y_cm)
# df_baseline['baseline_x_cm'] = pd.read_csv('cm.csv').iloc[:]['dot_x_cm']
# df_baseline['baseline_y_cm'] = pd.read_csv('cm.csv').iloc[:]['dot_y_cm']
df_baseline.to_csv('baseline.csv',index=False)
# print(df_tobii.head())
##################Compare df_baseline with df_tobii####################################
# loop through baseline timestamps
# segment to 15 sections
num_of_seg = 15
# count = 0
min_timestamp_per_dot = []
max_timestamp_per_dot = []
index_of_start_per_dot = []
for i in range(1,len(df_baseline.index)-1):
    if df_baseline.iloc[i][2] != df_baseline.iloc[i+1][2]:
        # count += 1
        index_of_start_per_dot.append(i+1)
# print(index_of_start_per_dot)
index_of_start_per_dot = [0]+index_of_start_per_dot

# print(index_of_start_per_dot)
for i in range(len(index_of_start_per_dot)-1):
    min_timestamp = min(df_baseline.iloc[index_of_start_per_dot[i]:index_of_start_per_dot[i+1]]['record_timestamp_diff'])
    min_timestamp_per_dot.append(min_timestamp)
    max_timestamp = max(df_baseline.iloc[index_of_start_per_dot[i]:index_of_start_per_dot[i + 1]]['record_timestamp_diff'])
    max_timestamp_per_dot.append(max_timestamp)

# Above extrude the last dot, which will do below
index_of_start_last_dot = index_of_start_per_dot[-1] # 161892:
index_of_end_last_dot = len(df_baseline.index)
last_dot_min_timestamp = df_baseline.iloc[index_of_start_last_dot]['record_timestamp_diff']
last_dot_max_timestamp = df_baseline.iloc[index_of_end_last_dot-1]['record_timestamp_diff']

min_timestamp_per_dot = min_timestamp_per_dot + [last_dot_min_timestamp]
max_timestamp_per_dot = max_timestamp_per_dot + [last_dot_max_timestamp]

match_tobii_timestamp = []
match_baseline_timestamp = []
for i in range(len(min_timestamp_per_dot)):
    for j in range(len(df_tobii.index)):
        if min_timestamp_per_dot[i]<df_tobii.iloc[j]['tobii_timestamp']<max_timestamp_per_dot[i]:
            match_tobii_timestamp.append(df_tobii.iloc[j]['tobii_timestamp'])
            match_baseline_timestamp.append(min_timestamp_per_dot[i])
# print(match_baseline_timestamp)
print(len(match_tobii_timestamp))
df_final = pd.DataFrame(list(zip(match_tobii_timestamp, match_baseline_timestamp)), \
                           columns=['tobii_timestamp_min','baseline_timestamp'])
df_final = df_final.set_index('tobii_timestamp_min')
df_tobii = df_tobii.set_index('tobii_timestamp')
df_final2 = df_final.join(df_tobii, lsuffix='_left', rsuffix='_right')
df_final2 = df_final2.set_index('baseline_timestamp')
df_baseline = df_baseline.set_index('record_timestamp_diff')
df_final3 = df_final2.join(df_baseline, lsuffix='_left', rsuffix='_right')
df_final3 = df_final3.rename(columns = {'Unnamed': 'tobii_timestamp_min'})
df_final3.to_csv('matched_with_tobii_with_real_cm.csv',index=True, index_label='tobii_timestamp_min')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
from matplotlib.patches import Ellipse
df_all = pd.read_csv('matched_with_tobii_with_real_cm.csv')[:2940] # only first loop
baseline_x_cm = df_all.iloc[:]['baseline_x_cm'].tolist()
baseline_y_cm = df_all.iloc[:]['baseline_y_cm'].tolist()
tobii_x = df_all.iloc[:]['2d_x'].tolist()
tobii_y = df_all.iloc[:]['2d_y'].tolist()

f_record = open('world_cm.txt')
line = f_record.readlines()
f_record.close()
lines = []
[lines.append(line) for line in line]
world_x_cm = lines[0]
world_y_cm = lines[1]

baseline_x_rescaled = [float(baseline_x_cm[i])/float(world_x_cm) for i in range(len(baseline_x_cm))]
baseline_y_rescaled = [float(baseline_y_cm[i])/float(world_y_cm) for i in range(len(baseline_y_cm))]

###################################################
### Extract dots that within range of 20 degree ###
###################################################
human_to_screen = 120 # cm
degree_for_accuracy = 20 # degree
# radius in cm in the screen that within 20 degree
radius_20_degree_cm = 120*math.tan(degree_for_accuracy * math.pi / 180)

radius_x = radius_20_degree_cm/float(world_x_cm)
radius_y = radius_20_degree_cm/float(world_y_cm)


# figure out the range, where
mean = [0.5,0.5]
width = 2*radius_x
height = 2*radius_y
ell = mpl.patches.Ellipse(xy=mean, width=width, height=height,facecolor='none',edgecolor='gray',linewidth=4)
fig, ax = plt.subplots()
ax.set_facecolor('black')
ax.set_xticks([])
ax.set_yticks([])
ax.add_patch(ell)

# plt.scatter(tobii_x,tobii_y)
plt.scatter(baseline_x_rescaled, baseline_y_rescaled,color='white')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Scene from Tobii's Scene Camera in x axis",fontsize = 14)
plt.ylabel("Scene from Tobii's Scene Camera in y axis",fontsize = 14)
plt.legend(['Range for accuracy assessment', 'Baseline dot'],fontsize = 14)
plt.savefig('tobii_baseline_20_degree.pdf',dpi=1200)
plt.show()

# which dots are within 20 degree
index_within_20_degree = []
for row in range(df_all.shape[0]):
     if (df_all.iloc[row]['baseline_x'] == 4 or df_all.iloc[row]['baseline_x'] == 5) and df_all.iloc[row]['baseline_y'] == 0:
         index_within_20_degree.append(row)
     elif (df_all.iloc[row]['baseline_x'] == 3 or df_all.iloc[row]['baseline_x'] == 4 or df_all.iloc[row]['baseline_x'] == 5\
             or df_all.iloc[row]['baseline_x'] == 6) and df_all.iloc[row]['baseline_y'] == 2:
         index_within_20_degree.append(row)
     elif (df_all.iloc[row]['baseline_x'] == 2 or df_all.iloc[row]['baseline_x'] == 3 or df_all.iloc[row]['baseline_x'] == 4 or df_all.iloc[row]['baseline_x'] == 5 \
          or df_all.iloc[row]['baseline_x'] == 6) and df_all.iloc[row]['baseline_y'] == 4:
         index_within_20_degree.append(row)
     elif (df_all.iloc[row]['baseline_x'] == 3 or df_all.iloc[row]['baseline_x'] == 4 or df_all.iloc[row]['baseline_x'] == 5 \
          or df_all.iloc[row]['baseline_x'] == 6) and df_all.iloc[row]['baseline_y'] == 6:
         index_within_20_degree.append(row)
     elif (df_all.iloc[row]['baseline_x'] == 3 or df_all.iloc[row]['baseline_x'] == 4 or df_all.iloc[row]['baseline_x'] == 5)\
             and df_all.iloc[row]['baseline_y'] == 8:
         index_within_20_degree.append(row)

df_acc_20_degree = pd.DataFrame(columns=['tobii_timestamp_min','2d_x','2d_y','baseline_timestamp_diff','baseline_x','baseline_y','baseline_x_cm','baseline_y_cm'])
for i in range(len(index_within_20_degree)):
    for j in range(df_acc_20_degree.shape[1]):
        df_acc_20_degree.loc[i] = df_all.iloc[index_within_20_degree[i]][:]
df_acc_20_degree.to_csv('within_20_degree.csv',index=False)

# plt.scatter(df_acc_20_degree.iloc['baseline_x'][:].tolist(),df_acc_20_degree.iloc['baseline_y'][:].tolist())
# plt.show()

######################################
############# Accuracy ###############
######################################
screen_width_in_cm = 154 #cm
screen_height_in_cm = 87 #cm

# only read data within 20 degree
df_all = pd.read_csv('within_20_degree.csv')
baseline_x_cm = df_all.iloc[:]['baseline_x_cm'].tolist()
baseline_y_cm = df_all.iloc[:]['baseline_y_cm'].tolist()
tobii_x = df_all.iloc[:]['2d_x'].tolist()
tobii_y = df_all.iloc[:]['2d_y'].tolist()


baseline_x_rescaled = [float(baseline_x_cm[i])/float(world_x_cm) for i in range(len(baseline_x_cm))]
baseline_y_rescaled = [float(baseline_y_cm[i])/float(world_y_cm) for i in range(len(baseline_y_cm))]

#rescale coordinates to our system
vec_baseline_dots = np.zeros((len(baseline_x_rescaled),3))
vec_human = np.zeros((len(baseline_x_rescaled),3))
vec_tobii_dots = np.zeros((len(tobii_x),3))
angle_difference = []
import vg
import math

#calculate accuracy in terms of degree
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

for i in range(len(baseline_x_rescaled)):
    vec_baseline_dots[i,:]=[baseline_x_rescaled[i]*screen_width_in_cm-screen_width_in_cm/2,screen_height_in_cm/2-baseline_y_rescaled[i]*screen_height_in_cm,120]
    vec_tobii_dots[i, :] = [tobii_x[i]*screen_width_in_cm-screen_width_in_cm/2,screen_height_in_cm/2-tobii_y[i]*screen_height_in_cm,120]
    angle_difference.append((angle(vec_baseline_dots[i,:],vec_tobii_dots[i,:])))
accuracy_degree = sum(angle_difference)/len(angle_difference)
print('The accuracy of Tobii Glass in degree is {}'.format(accuracy_degree))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
# only read data within 20 degree
df_all = pd.read_csv('within_20_degree.csv')
baseline_x_cm = df_all.iloc[:]['baseline_x_cm'].tolist()
baseline_y_cm = df_all.iloc[:]['baseline_y_cm'].tolist()
tobii_x = df_all.iloc[:]['2d_x'].tolist()
tobii_y = df_all.iloc[:]['2d_y'].tolist()
baseline_x_coor = df_all.iloc[:]['baseline_x'].tolist()
baseline_y_coor = df_all.iloc[:]['baseline_y'].tolist()

f_record = open('world_cm.txt')
line = f_record.readlines()
f_record.close()
lines = []
[lines.append(line) for line in line]
world_x_cm = lines[0]
world_y_cm = lines[1]

baseline_x_rescaled = [float(baseline_x_cm[i])/float(world_x_cm) for i in range(len(baseline_x_cm))]
baseline_y_rescaled = [float(baseline_y_cm[i])/float(world_y_cm) for i in range(len(baseline_y_cm))]

#rescale coordinates to our system
vec_baseline_dots = np.zeros((len(baseline_x_rescaled),3))
vec_human = np.zeros((len(baseline_x_rescaled),3))
vec_tobii_dots = np.zeros((len(tobii_x),3))
angle_difference = []
import vg
import math
######################################
############# Accuracy ###############
######################################
screen_width_in_cm = 154 #cm
screen_height_in_cm = 87 #cm

#calculate accuracy in terms of degree
def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

for i in range(len(baseline_x_rescaled)):
    vec_baseline_dots[i,0]=baseline_x_rescaled[i]*screen_width_in_cm-screen_width_in_cm/2
    vec_baseline_dots[i, 1] = screen_height_in_cm/2-baseline_y_rescaled[i]*screen_height_in_cm
    vec_baseline_dots[i, 2] = 120
    vec_tobii_dots[i, 0] = tobii_x[i]*screen_width_in_cm-screen_width_in_cm/2
    vec_tobii_dots[i, 1] = screen_height_in_cm/2-tobii_y[i]*screen_height_in_cm
    vec_tobii_dots[i, 2] = 120
    angle_difference.append((angle(vec_baseline_dots[i,:],vec_tobii_dots[i,:])))
accuracy_degree = sum(angle_difference)/len(angle_difference)
print('The accuracy of Tobii Glass in degree is {}'.format(accuracy_degree))


######################################
############# Precision ###############
######################################
angle_predict_to_human = []

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
for i in range(len(baseline_x_coor)-1):
    if baseline_x_rescaled[i+1]-baseline_x_rescaled[i]>0:
        index_of_each_dot.append(i+1)
index_of_each_dot = index_of_each_dot + [len(baseline_x_rescaled)]
print(len(index_of_each_dot))
# for i in range(index_of_each_dot[0],index_of_each_dot[1]):
#     vec_predict_dots[i, 0] = poly_pred[i] * times_screen_width - 154 / 2
#     vec_predict_dots[i, 1] = poly_pred2[i] * times_screen_height - 87 / 2
#     vec_predict_dots[i, 2] =120
for j in range(len(index_of_each_dot)-1):
    for i in range(index_of_each_dot[j],index_of_each_dot[j+1]):
        vec_tobii_dots[i, 0] = tobii_x[i] * screen_width_in_cm - screen_width_in_cm / 2
        vec_tobii_dots[i, 1] = screen_height_in_cm / 2 - tobii_y[i] * screen_height_in_cm
        vec_tobii_dots[i, 2] = 120
        vec_human[i, 0] = 0
        vec_human[i, 1] = 0
        vec_human[i, 2] = 0

        # angle_predict_to_human.append((angle(vec_predict_dots[i-1,:],vec_predict_dots[i,:])))
for i in range(1,len(baseline_x_rescaled)):
    angle_predict_to_human.append(math.degrees(angle_between(vec_tobii_dots[i-1,:],vec_tobii_dots[i,:])))
# precision = sum(angle_predict_to_human[1:])/len(angle_predict_to_human[1:])
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
print(precision_rms)
