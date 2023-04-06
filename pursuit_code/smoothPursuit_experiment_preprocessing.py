import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import time

#################
# USER SETTINGS #
#################
directory = 'pursuit'
plot_data = True

######################
# INTERNAL VARIABLES #
######################
dir_char = '/'
files = []
file_times = {} # key: file_time, value: list of files with this file_time
output_df = pd.DataFrame(columns=['LL', 'LC', 'LR', 'RL', 'RC', 'RR', 'dot_x', 'dot_y', 'sensor_timestamp', 'dot_timestamp', 'record_timestamp', 'movement_number'])
sensor_colors = ['orange', 'teal', 'mediumorchid', 'lightcoral', 'deepskyblue', 'blue']
line_colors = ['red', 'maroon']

#######
# RUN #
#######
# start timer
start_timer = time.time()

# reset dir_char if running on Windows
if sys.platform in ['Win32', 'Win64']:
    dir_char = '\\'

# populate file_times
files = os.listdir(directory)
for file in files:
    # skip any files that aren't csv files
    if file[-4:] != '.csv':
        continue
    # ask user what to do if preprocessed data file already exists
    if 'preprocessed' in file:
        user_input = input("\nThe file '%s' already exists, replace it? 'yes' or 'quit': " % file)
        if user_input.lower()[0] != 'y':
            print('\nGoodbye.')
            quit()
        else:
            os.remove(directory + dir_char + file)
            print("'%s' deleted." % (file))
    else:
        file_time = file.split('_')[6][:-4]
        if file_time not in file_times.keys():
            file_times[file_time] = [file]
        else:
            file_times[file_time].append(file)

# Message user
print("\nProcessing %d dot/sensor dataset pairs in directory '%s':\n" % (len(file_times), directory))

# Process all line dataset pairs
for file_time_index, file_time in enumerate(file_times.keys()):

    # get dot/sensor dataset pair for a single dot line
    line_dataPair = sorted(file_times[file_time])

    # message user with update
    print('%d/%d: processing files recorded %s:' % (file_time_index+1, len(file_times), file_time))
    print('        %s\n        %s' % tuple(line_dataPair))

    # if dataset missing: error message then quit
    if len(line_dataPair) != 2:
        print('Error: a dataset is missing for the timestamp:', file_time)
        quit()

    # import files, convert datetime columns
    dot_df = pd.read_csv(directory + dir_char + line_dataPair[0])
    sensor_df = pd.read_csv(directory + dir_char + line_dataPair[1])
    dot_df['timestamp']  = pd.to_datetime(dot_df['timestamp'])
    sensor_df['timestamp']  = pd.to_datetime(sensor_df['timestamp'])

    # remove sensor data that is outside dot data range
    start_time = dot_df['timestamp'].min()
    stop_time = dot_df['timestamp'].max()
    sensor_df = sensor_df[(sensor_df['timestamp'] >= start_time) & (sensor_df['timestamp'] <= stop_time)]

    # get dot start_stop values
    dot_x_start, dot_y_start, dot_x_stop, dot_y_stop = [float(i) for i in line_dataPair[0].split('_')[2:6]]

    # choose which dot axes will be used for parsing line movements (the one with biggest range)
    track_col = []
    if (abs(dot_x_start - dot_x_stop) > abs(dot_y_start - dot_y_stop)):
        track_col = ['dot_x', dot_x_start, dot_x_stop]
    else:
        track_col = ['dot_y', dot_y_start, dot_y_stop]

    # get start/stop time points of each line movement
    lineMov_endpoints = []
    endval_seek = 2 # an index of track_col. Used to flip whether searching for dot start or dot stop value.
    # 1/2: get stop points for each line movement: looks for 'lead-in' to final dot position of line movement
    for i in range(1, len(dot_df)):
        if ((dot_df.loc[i, track_col[0]] == track_col[endval_seek]) and (dot_df.loc[i-1, track_col[0]] != track_col[endval_seek])):
            lineMov_endpoints.append([i])
            endval_seek = (endval_seek % 2) + 1
    # 2/2: make a list of start/stop indexes for each line movement by adding start points
    for i in range(len(lineMov_endpoints)):
        if i == 0:
            lineMov_endpoints[0].insert(0, 0)
        else:
            lineMov_endpoints[i].insert(0, lineMov_endpoints[i-1][1] + 1)

    # add dot position and timestamp columns to a sensor_df
    for n, line_movement in enumerate(lineMov_endpoints):
        # df of a single line movement
        temp_dot_df = dot_df.loc[line_movement[0]:line_movement[1]].reset_index()
        for i in range(1, len(temp_dot_df)):
            # get time range of each dot position, use it to select a sensor df, add needed columns, concat to output_df
            ts_vals = temp_dot_df.loc[i-1:i, 'timestamp'].values
            temp_sensor_df = sensor_df[sensor_df['timestamp'].between(*ts_vals, 'left')].copy()
            temp_sensor_df['dot_x'] = temp_dot_df.loc[i-1, 'dot_x']
            temp_sensor_df['dot_y'] = temp_dot_df.loc[i-1, 'dot_y']
            temp_sensor_df['dot_timestamp'] = temp_dot_df.loc[i-1, 'timestamp']
            temp_sensor_df['record_timestamp'] = file_time
            temp_sensor_df['movement_number'] = n
            temp_sensor_df.rename(columns = {'timestamp':'sensor_timestamp'}, inplace = True)
            output_df = pd.concat([output_df, temp_sensor_df], sort=False)

# save processed data as csv, notify user
save_filename = directory + dir_char + directory + '_preprocessed.csv'
output_df.to_csv(save_filename, header=True, index=False)
time_taken = time.time() - start_timer
print("\n'%s' created.\nTime taken: %d minutes %d seconds." % (save_filename, int(time_taken/60), int(time_taken % 60)))

######################
# RE_IMPORT AND PLOT #
######################
if plot_data:

    print('\nRe-importing/plotting data . . .')

    # import data
    plot_df = pd.read_csv(save_filename)
    plot_df['sensor_timestamp']  = pd.to_datetime(plot_df['sensor_timestamp'])
    plot_df['dot_timestamp']  = pd.to_datetime(plot_df['dot_timestamp'])

    # create figure
    fig = plt.figure()
    axes = {}

    # add subplots and data
    for i, record_time in enumerate(plot_df['record_timestamp'].unique()):
        data = plot_df[plot_df['record_timestamp'] == record_time]
        axes_dimensions = math.ceil(math.sqrt(len(plot_df['record_timestamp'].unique())))
        axes['%d_sensor' % i] = fig.add_subplot(axes_dimensions, axes_dimensions, i+1)
        axes['%d_dot' % i] = axes['%d_sensor' % i].twinx()
        axes['%d_sensor' % i].axes.xaxis.set_visible(False)
        axes['%d_sensor' % i].axes.yaxis.set_visible(False)
        axes['%d_dot' % i].axes.yaxis.set_visible(False)
        axes['%d_dot' % i].plot(data['dot_timestamp'], data['dot_x'], color='red', linewidth=4)
        axes['%d_dot' % i].plot(data['dot_timestamp'], data['dot_y'], color='maroon', linewidth=2.5)
        axes['%d_dot' % i].plot(data['dot_timestamp'], data['dot_x'], color=line_colors[0])
        axes['%d_dot' % i].plot(data['dot_timestamp'], data['dot_y'], color=line_colors[1])
        for j, sensor in enumerate(['LL', 'LC', 'LR', 'RL', 'RC', 'RR']):
            axes['%d_sensor' % i].plot(data['sensor_timestamp'], data[sensor], color=sensor_colors[j], label=sensor + '_sensor')

    # legend
    axes['0_sensor'].plot([],[], color=line_colors[0], label='dot_x')
    axes['0_sensor'].plot([],[], color=line_colors[1], label='dot_y')
    axes['0_sensor'].legend(loc='lower left', ncol=4,  bbox_to_anchor=[0, 1.1], fontsize=8)

    # format and show figure
    fig.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
