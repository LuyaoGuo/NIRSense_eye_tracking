# This code is for recording eye tests with a 6-sensor eye tracking system.
# It presents the user with a dot that moves back and forth along a line
# for the user to follow with their eyes, stopping at each end of the line until
# the user presses Enter to continue (the start/end coordinates of the line can
# be set below).
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
#       (carriage return is optional.)
#
# After the test, the main process re-opens the saved files and plots them to confirm
# the data was recorded as intended.
# Several settings can be changed: see the USER SETTINGS section below for more
# details.
#
# Data files are saved in the directory:
#
#                                ./test_data
#
# It will be created if it doesn't exist. Note if on Linux: this code must be run
# by superuser so superuser will also be needed to delete this directory,
# as well the files created by this program.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
import serial
import sys
from multiprocessing import Process, Pipe
from datetime import datetime
import keyboard
import time
import pandas as pd
import signal

#################
# USER SETTINGS #

#################
num_repeat = 1
# number of times the dot will move forward and backwards
num_dot_positions = 150        # Number of dot positions along the line (also determines speed).
dot_start_x = 0                # dot start and stop locations. Should be between 0 and axes_lim.
dot_stop_x = 8

dot_start_y = 4
dot_stop_y = 4

confirm_data = True             # Boolean. Display saved data after the test.
record_sensors = True           # Boolean. Record sensors (or just dot data).
process_sensor_data = True      # Boolean. Validate sensor data/save as csv, or: don't validate, save as .txt
serial_port = 'COM8'
baud_rate = 1000000
dir_char = '/'                  # Directory character for filepath, will be automatically reset for Windows.
# set and forge
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
user_input = False               # stores user input (True), should be reset to False at each side of screen.
dot_coords = np.empty((0, 0))    # contains all dot coords used in test.
coords_index = 0
dot_df = pd.DataFrame(columns=dot_df_columns)
rec_stop_points = []
stop_point_i = 0
anim_setup = 0                   # Sets return after first anim call, so plot is drawn while waiting for user input. Init to True.
setup_frames = 5                 # this istartup buffer is needed to display animation before starting test.
filepath = ""
dot_process = None              # Sub-processes that will be used.
sensor_process = None
sensor_filepath = ""
dot_PID = None
sensor_PID = None
#############
# FUNCTIONS #
#############

# Handle SIGINT (Ctrl-C from user, used to exit)
def SIGINT_handler(signum, frame):
    global dot_process, sensor_process, record_sensors, dot_PID, sensor_PID

    if dot_process.is_alive():
        dot_process.terminate()
    if record_sensors:
        if sensor_process.is_alive():
            sensor_process.terminate()
    print('\n  Processes terminated. Goodbye.\n')
    exit()

# Re-import and plot data files after test to check data was recorded correctly.
def post_test_check(dot_filepath, sensor_filepath):

    subplots = {}

    #import dot dataset
    dotCheck_df = pd.read_csv(dot_filepath)
    dotCheck_df['timestamp'] = pd.to_datetime(dotCheck_df['timestamp'])

    # check if sensors were recorded
    n_subplots = 2
    if "csv" in sensor_filepath:
        n_subplots = 4
        sensorCheck_df = pd.read_csv(sensor_filepath)
        sensorCheck_df['timestamp'] = pd.to_datetime(sensorCheck_df['timestamp'])
    # create plots
    fig = plt.figure()
    for i in range(n_subplots):
        subplots[i] = fig.add_subplot(2 ,2, i+1)
    # Finalise dot data plots
    subplots[0].set_title('Dot Data', fontsize=16, fontweight='bold')
    subplots[1].set_title('Dot Data FPS', fontsize=16, fontweight='bold')
    subplots[0].scatter(dotCheck_df[dot_df_columns[0]], dotCheck_df[dot_df_columns[1]], label=dot_df_columns[1])
    subplots[0].scatter(dotCheck_df[dot_df_columns[0]], dotCheck_df[dot_df_columns[2]], label=dot_df_columns[2])
    subplots[1].plot(dotCheck_df['timestamp'], [1000000/time.microseconds for time in dotCheck_df['timestamp'].diff()])
    subplots[1].set_xlim(subplots[0].get_xlim())
    subplots[0].legend()
    subplots[0].grid()
    subplots[1].grid()
    # sensor plots
    if n_subplots == 4:
        subplots[2].set_title('Sensor Data', fontsize=16, fontweight='bold')
        subplots[3].set_title('Sensor Sample Freq', fontsize=16, fontweight='bold')
        for sensor in sensorCheck_df.columns[1:]:
            subplots[2].plot(sensorCheck_df['timestamp'], sensorCheck_df[sensor], label = sensor)
        subplots[3].plot(sensorCheck_df['timestamp'], [1000000/time.microseconds for time in sensorCheck_df['timestamp'].diff()])
        subplots[2].legend()
        subplots[2].set_xlim(subplots[0].get_xlim())
        subplots[3].set_xlim(subplots[0].get_xlim())
        subplots[2].grid()
        subplots[3].grid()
        subplots[2].set_ylabel('mV')
        subplots[3].set_ylabel('Hz')

        fig.tight_layout()
    # format and show figure
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


# MAIN FUNCTION OF DOT PROCESS
def dot_run(dot_to_main, dot_to_sensor, dot_from_sensor):
    global dot_coords, record_sensors

    # send PID
    dot_to_main.send(os.getpid())

    ### PROCESS FUNCTIONS ###

    def save_dot_csv():
        # save .csv file
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        savetime = datetime.now().strftime('%Y-%m-%dT%H%M%S')
        filename = 'dot_data_%d_%d_%d_%d_%s.csv' % (dot_start_x, dot_start_y, dot_stop_x, dot_stop_y, savetime)
        dot_df.to_csv(data_folder + dir_char + filename, header=True, index=False)

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
            dot_to_sensor.send(filepath)
            if confirm_data:
                exit_message = "Test Completed.\n\n" +\
                          ("Dot data saved as:\n%s\n\n" % filepath) +\
                          "Press Enter to close window and view data . . .\n(may take some time)"
            else:
                exit_message = "Test Completed\n\n" +\
                               "Data confirmation turned off.\n\n" +\
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
        if coords_index < dot_coords.shape[0]:
            dot_df.loc[coords_index] = [datetime.now(), dot_coords[coords_index][0], dot_coords[coords_index][1]]

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
        point.set_data([dot_coords[coords_index][0]],[dot_coords[coords_index][1]])
        return [point]

    ### END dot_run PROCESS FUNCTION DEFINITIONS ###

    # populate dot_coords with all required dot x and y coordinates
    for i, dot_axis in enumerate([[dot_start_x, dot_stop_x], [dot_start_y, dot_stop_y]]):
        # create complete dot coordnates for dot_axis (x or y)
        coords_start = np.linspace(dot_axis[0], dot_axis[0], int(num_dot_positions/4))
        coords_forward = np.linspace(dot_axis[0], dot_axis[1], num_dot_positions)
        coords_terminal = np.linspace(dot_axis[1], dot_axis[1], int(num_dot_positions/4))
        coords_backward = np.linspace(dot_axis[1], dot_axis[0], num_dot_positions)
        coords = np.concatenate([coords_start, coords_forward, coords_terminal, coords_backward])
        coords = np.tile(coords, num_repeat)
        # add to dot_coords
        dot_coords = dot_coords.reshape((i, len(coords)))
        dot_coords = np.vstack((dot_coords, coords))
    # transpose dot_coords
    dot_coords = dot_coords.T

    # calculate recording stop points
    for i in range(num_repeat):
        repeat_interval = 2*int(num_dot_positions/4) + 2*num_dot_positions
        rec_stop_points.append(int(num_dot_positions/4) + num_dot_positions + i*repeat_interval)
        rec_stop_points.append(2*int(num_dot_positions/4) + 2*num_dot_positions + i*repeat_interval)

    # wait for sensor process to confirm connection
    if record_sensors:
        while not dot_from_sensor.poll():
            time.sleep(0.01)
        print("\n  %s\n" % dot_from_sensor.recv())

    # Create and initialise animation plot
    fig = plt.figure(facecolor='black')
    axes = fig.add_subplot(111, autoscale_on=False)
    axes.set_xlim(0-0.01*axes_lim, 8+0.01*axes_lim)
    axes.set_ylim(0-0.01*axes_lim, 8+0.01*axes_lim)
    axes.get_xaxis().set_ticks([])
    axes.get_yaxis().set_ticks([])
    fig.set_size_inches(18.5, 10.5, forward=True)
    axes.set_facecolor('black')
    point, = axes.plot([dot_start_x],[dot_start_y], 'wo', animated=blit, markersize=markersize)
    message = axes.text(axes_lim/2, axes_lim/2, '', fontsize=20, fontweight='bold', color='white', ha='center', va='center', zorder=10)

    # create animation
    animation = FuncAnimation(fig, ani, interval=200, blit=blit, repeat=False)
    # format display
    fig.tight_layout() #reduce margins
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    # Animation over: send filepath, finish process
    dot_to_main.send(filepath)
    return

# MAIN FUNCTION OF SENSOR PROCESS
def sensor_record(sensor_from_dot, sensor_to_main, sensor_to_dot):

    # send PID
    sensor_to_main.send(os.getpid())

    ### PROCESS VARIABLES ###
    data = []
    timestamps = []

    # create serial connection
    print('\n  Connecting to serial device . . .')
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    # read lines until reading syncs, notify dot_process when connected.
    header_i = 200
    while header_i > 0:
        raw_data = ser.readline()
        header_i = header_i - 1
    # Confirm stable connection with dot_process
    sensor_to_dot.send("Serial device connected.")

    # read data until signalled to stop
    while not sensor_from_dot.poll():
        try:
            raw_data = ser.readline()
            raw_data = raw_data.decode()
            data.append(raw_data)
            timestamps.append(datetime.now())
        except:
            pass

    # message user, get filepath
    print('\n  Processing data . . .')
    dot_filepath = sensor_from_dot.recv()
    sensor_filepath = dot_filepath.replace('dot', 'sensor')

    # Process data: check for valid lines by tring to convert strings to float.
    if process_sensor_data:
        file = open(sensor_filepath, 'w')
        file.write("timestamp,LL,LC,LR,RL,RC,RR\n")
        for i, ts in enumerate(timestamps):
            try:
                line = data[i].replace('\n', "").replace('\r', "").split(',')
                line = "%f,%f,%f,%f,%f,%f" % tuple([float(x) for x in line])
                file.write(ts.strftime('%Y-%m-%dT%H:%M:%S.%f') + ',' + line + "\n")
            except:
                pass
        file.close()

    # Else data processing turned off: just save raw data text file
    else:
        sensor_filepath = sensor_filepath.replace("csv", "txt")
        file = open(sensor_filepath, 'w')
        for i, ts in enumerate(timestamps):
            line = data[i].replace('\n', "").replace('\r', "")
            file.write(ts.strftime('%Y-%m-%dT%H:%M:%S.%f') + ',' + line + "\n")
        file.close()

    # Data recording and processing finished, signal main process and return
    print('  Data processing completed.\n  Sensor data saved as:\n  %s\n' % sensor_filepath)
    sensor_to_main.send(sensor_filepath)
    return

#######
# RUN #
#######
if __name__=='__main__':

    # reset dir_char if running on Windows
    if sys.platform in ['Win32', 'Win64']:
        dir_char = '\\'

    # Process comms
    main_from_dot, dot_to_main = Pipe()
    sensor_from_dot, dot_to_sensor = Pipe()
    main_from_sensor, sensor_to_main = Pipe()
    dot_from_sensor, sensor_to_dot = Pipe()

    # Create and start processes
    dot_process = Process(target=dot_run, args=(dot_to_main, dot_to_sensor, dot_from_sensor,))
    dot_process.start()
    if record_sensors:
        sensor_process = Process(target=sensor_record, args=(sensor_from_dot, sensor_to_main, sensor_to_dot,))
        sensor_process.start()

    # setup SIGINT_handler: exit using ctrl-C and it will terminate processes.
    signal.signal(signal.SIGINT, SIGINT_handler)

    # get and print PIDs
    sensor_PID_string = ""
    dot_PID = main_from_dot.recv()
    if record_sensors:
        sensor_PID = main_from_sensor.recv()
        sensor_PID_string = "\n  sensor_record process started - PID: %d" % sensor_PID
    print("\n  dot_run process started - PID: %d%s" % (dot_PID, sensor_PID_string))

    # wait for confirmation of data file creation from sub-processes
    dot_filepath = main_from_dot.recv()
    print('Hey')
    if record_sensors:
        sensor_filepath = main_from_sensor.recv()

    # terminate sub-processes after all inter-process comms are done
    if record_sensors:
        sensor_process.terminate()
    dot_process.terminate()
    print("  (Sub-processes successfully terminated.)\n")

    # confirm data
    if confirm_data:
        post_test_check(dot_filepath, sensor_filepath)
