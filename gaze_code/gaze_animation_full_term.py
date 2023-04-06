# This code is for recording eye tests with a 6-sensor eye tracking system.
# It presents the user with a freezing dot that moves to cover the screen
# for the user to follow with their eyes, staring at the dot each time
# with long press 'right' button during the period of staring for recording
# press 'enter' button to move the freezing dot to another position
#
# Two sub-processes are created:
# 1) dot_process:
#       Handles the animation, dot data recording, user input. Saves timestamped
#       dot coordinates in the dot_df dataframe then as a csv file before returning.
# 2) sensor_process:
#       records sensor data from a serial connection. Stores raw data as strings
#       for faster sampling, then when test is completed, can verify each line as
#       valid before saving as a csv file. The expected line is 6 comma-separated
#       numeric values, eg:
#
#           "<float>,<float>,<float>,<float>,<float>,<float>\r\n"
#
# Several settings can be changed: see the USER SETTINGS section below for more details.
#
# Data files are saved in the same directory
import winsound

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import serial
import sys
from multiprocessing import Process, Pipe
import multiprocessing
from datetime import datetime
import keyboard
import time
import pandas as pd
import signal
from time import sleep


#################
# USER SETTINGS #
#################
# num_repeat is the number of repeating times the dot will loop through over the screen
num_repeat = 3

# Number of dot positions along each row
num_dot_positions = 9

# dot start and stop locations. Should be between 0 and axes_lim.
dot_start_x = 0
dot_stop_x = 8
dot_start_y = 8
dot_stop_y = 8

confirm_data = True  # Boolean. Display saved data after the test.
record_sensors = True  # Boolean. Record sensors (or just dot data).
process_sensor_data = True  # Boolean. Validate sensor data/save as csv, or: don't validate, save as .txt

# Information about the serial connection
serial_port = 'COM3'
baud_rate = 1000000
dir_char = '/'  # Directory character for filepath, will be automatically reset for Windows.
# set and forget
blit = True
markersize = 7
axes_lim = 8
data_folder = 'test_data'
dot_df_columns = ['timestamp', 'dot_x', 'dot_y']
sensor_df_columns = ['timestamp', 'LL', 'LC', 'LR', 'RL', 'RC', 'RR']

######################
# INTERNAL VARIABLES #
######################
exit_state = 0
user_input = False  # stores user input (True), should be reset to False at each side of screen.
dot_coords = np.empty((0, 0))  # contains all dot coords used in test.
coords_index = 0
dot_df = pd.DataFrame(columns=dot_df_columns)
rec_stop_point = np.arange(1, (9 * 5 * num_repeat) + 1, 1)
rec_stop_points = rec_stop_point.tolist()

stop_point_i = 0
anim_setup = 0  # Sets return after first anim call, so plot is drawn while waiting for user input. Init to True.
setup_frames = 5  # this istartup buffer is needed to display animation before starting test.
filepath = "test"
dot_process = None  # Sub-processes that will be used.

sensor_process = None
sensor_filepath = ""
dot_PID = None
sensor_PID = None

# Specify the file name by the time taking the experiment
file_time = datetime.now()
file_naming_format = '{}_{}_{}_{}_{}'.format(dot_start_x, dot_start_y, dot_stop_x, dot_stop_y,
                                             file_time.strftime("%Y_%m_%d_%H_%M_%S"))


#############
# FUNCTIONS #
#############


# MAIN FUNCTION OF DOT PROCESS
def dot_run():
    global dot_coords

    ### PROCESS FUNCTIONS ###

    def save_dot_csv():
        # save .csv file
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        savetime = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        filename = 'dot_data_%d_%d_%d_%d_%s.csv' % (dot_start_x, dot_start_y, dot_stop_x, dot_stop_y, savetime)
        # dot_df.to_csv(data_folder + dir_char + filename, header=True, index=False)

        return data_folder + dir_char + filename

    # animation loop function
    def ani(frame):
        global exit_state, user_input, coords_index, dot_coords, coords_index
        global rec_stop_points, stop_point_i, anim_setup, setup_frames, filepath

        # takes a few frames to set up figure
        if frame < setup_frames:
            return [point, message]

        # EXIT SEQUENCE
        # 1) save data, signal sensor process to stop, display exit message.
        if exit_state == 1:
            filepath = save_dot_csv()
            if confirm_data:
                exit_message = "Test Completed.\n\n" + \
                               ("Dot data saved as:\n%s\n\n" % filepath) + \
                               "Press Enter to close window and view data . . .\n(may take some time)" + \
                               "After window is closed, press 'Left' keyboard to end the entire calibration"
            else:
                exit_message = "Test Completed\n\n" + \
                               "Data confirmation turned off.\n\n" + \
                               "Press Enter to Quit."
            message.set_text(exit_message)
            exit_state = exit_state + 1
            point.set_color('white')
            return [point, message]

        # 2) wait for user confirmation then close animation
        if exit_state == 2:
            keyboard.wait('Enter')
            plt.close()
            return [point, message]

        # DURING TEST: wait for user input
        if user_input == False:
            keyboard.wait('Enter')
            user_input = True
            point.set_color('lime')
            return [point, message]

        # Record current dot coords (set on startup or by last loop).
        dot_file = 'dot_data_{}.csv'.format(file_naming_format)
        if coords_index < dot_coords.shape[0]:
            time = datetime.now()
            time_str = datetime.strftime(time, "%m/%d/%Y %H:%M:%S:%f")
            dot_df.loc[coords_index] = [time_str, dot_coords[coords_index][0], dot_coords[coords_index][1]]

            dot_df.to_csv(dot_file,index=False)

        # All dot coords used: trigger exit sequence.
        if coords_index == dot_coords.shape[0] - 1:
            exit_state = exit_state + 1
            return [point, message]

        # Dot has reached side of screen on last update: reset user_input and return.
        if coords_index in rec_stop_points:
            rec_stop_points.remove(coords_index)
            user_input = False
            point.set_color('white')
            stop_point_i = (stop_point_i + 1) % 2
            return [point, message]

        # Side of screen not reached: increment coords_index, update dot position, return.
        coords_index = coords_index + 1
        point.set_data([dot_coords[coords_index][0]], [dot_coords[coords_index][1]])
        return [point]

    ### END dot_run PROCESS FUNCTION DEFINITIONS ###
    x_axis = np.tile(np.linspace(0, 8, 9), 5)
    y_axis = np.concatenate(
        (np.linspace(8, 8, 9), np.linspace(6, 6, 9), np.linspace(4, 4, 9), np.linspace(2, 2, 9), np.linspace(0, 0, 9)))
    coords = np.vstack((x_axis, y_axis)).T
    dot_coords = coords
    dot_coords = np.tile(dot_coords, (num_repeat, 1))
    dot_coords = np.vstack([dot_coords, np.array((8, 0))])

    # calculate recording stop points
    for i in range(num_repeat):
        repeat_interval = 2 * int(num_dot_positions / 4) + 2 * num_dot_positions

    # Create and initialise animation plot
    fig = plt.figure(facecolor='black')
    axes = fig.add_subplot(111, autoscale_on=False)
    axes.set_xlim(0 - 0.01 * axes_lim, 8 + 0.01 * axes_lim)
    axes.set_ylim(0 - 0.01 * axes_lim, 8 + 0.01 * axes_lim)
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    fig.set_size_inches(18.5, 10.5, forward=True)
    axes.set_facecolor('black')
    point, = axes.plot([dot_start_x], [dot_start_y], 'wo', animated=blit, markersize=markersize)
    message = axes.text(axes_lim / 2, axes_lim / 2, '', fontsize=20, fontweight='bold', color='white', ha='center',
                        va='center', zorder=10)

    # create animation
    animation = FuncAnimation(fig, ani, interval=20, blit=blit, repeat=False)
    # format display
    fig.tight_layout()  # reduce margins

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    return exit_state



def save_data():
    global file_time, exit_state, serial_port, baud_rate

    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    sensor_file = 'sensor_data_{}.csv'.format(file_naming_format)
    file = open(sensor_file, 'w')
    # df_sensor
    file.write("timestamp,LL,LC,LR,RL,RC,RR\n")

    # Get data from the serial port
    while True:
        raw_data = ser.readline()
        try:
            data_str = raw_data.decode()

            if keyboard.is_pressed('right'):
                date = datetime.now()
                date_str = datetime.strftime(date, "%m/%d/%Y %H:%M:%S:%f,")
                file.write(date_str)
                file.write(data_str)
        except:
            pass


if __name__ == '__main__':
    p1 = Process(target=dot_run)
    p2 = Process(target=save_data)
    p1.start()
    p2.start()
