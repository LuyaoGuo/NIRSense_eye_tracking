import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###########################################
### Calculate dimension in pixel way ######
###########################################
df_tobii = pd.read_csv(r'C:\Users\lguo8\PycharmProjects\EyeTracking_baseline\Tobii\luyao_12_13_3_light_on\20221213T071219Z\cv2_pixels\pixel_coordinates.csv',header=None )
df_tobii.columns = ['tobii_pixel_x', 'tobii_pixel_y', 'baseline_x', 'baseline_y']

df_world_camera = pd.read_csv(r'C:\Users\lguo8\PycharmProjects\EyeTracking_baseline\Tobii\luyao_12_13_3_light_on\20221213T071219Z\cv2_pixels\pixel_world_camera.csv',header=None)
tobii_pixel_x = df_tobii.iloc[:]['tobii_pixel_x'].tolist()
tobii_pixel_y = df_tobii.iloc[:]['tobii_pixel_y'].tolist()
baseline_x = df_tobii.iloc[:]['baseline_x'].tolist()
baseline_y = df_tobii.iloc[:]['baseline_y'].tolist()
# plt.scatter(tobii_pixel_x, tobii_pixel_y)
# plt.show()

# define screen outlines pixels
# assume as rectangle

for i in range(len(baseline_x)):
    if baseline_x[i]==0 and baseline_y[i]== 8:
        index_left_top = i
    elif baseline_x[i]==0 and baseline_y[i]== 0:
        index_left_bottom = i
    elif baseline_x[i]==8 and baseline_y[i]== 0:
        index_right_bottom = i
    elif baseline_x[i]==8 and baseline_y[i]== 8:
        index_right_top = i
# print(index_left_top)
# due to the fact that pixel is defined by dot, a few space is saved for dot size
value_given = 6
# pixel in y axis is reverse to baseline to y axis

pixel_left_bottom = (tobii_pixel_x[index_left_bottom]-value_given, tobii_pixel_y[index_left_bottom]-value_given)
pixel_left_top = (tobii_pixel_x[index_left_top]-value_given, tobii_pixel_y[index_left_top]+value_given)
pixel_right_bottom = (tobii_pixel_x[index_right_bottom]+value_given, tobii_pixel_y[index_right_bottom]-value_given)
pixel_right_top = (tobii_pixel_x[index_right_top]+value_given, tobii_pixel_y[index_right_top]+value_given)

# use left_bottom, right_bottom and righttop to define screen outlines
x=[pixel_left_bottom[0],pixel_right_top[0],pixel_right_top[0],pixel_right_bottom[0]]
y=[pixel_left_bottom[1],pixel_right_top[1],pixel_right_top[1],pixel_right_bottom[1]]

# plt.plot(x,y, 'ro')

def connectpoints(x,y,p1,p2):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1,x2],[y1,y2],'k-')

# treat these 2 lines as screen outlines
plt.plot([x[0],x[3]],[y[0],y[3]],'k-')
plt.plot([x[2],x[3]],[y[2],y[3]],'k-')

# calculate slopes and intercepts
slope_bottom = (y[3]-y[0])/(x[3]-x[0])
intercept_bottom = y[3]-slope_bottom*x[3]

screen_left_top_x = x[0]+x[2]-x[3]
screen_left_top_y = y[0]+y[2]-y[3]
plt.plot([x[0],screen_left_top_x],[y[0],screen_left_top_y],'k-')
plt.plot([x[2],screen_left_top_x],[y[2],screen_left_top_y],'k-')


# define xlim and ylim by world_camera_pixels
world_x_min = df_world_camera.iloc[0][0]
world_x_max = df_world_camera.iloc[2][0]
world_y_min = df_world_camera.iloc[0][1]
world_y_max = df_world_camera.iloc[2][1]
plt.xlim([world_x_min, world_x_max])
plt.ylim([world_y_min, world_y_max])
# plt.axis('equal')
plt.show()


screen_x=[screen_left_top_x,pixel_left_bottom[0],pixel_right_top[0],pixel_right_bottom[0]]
screen_y=[screen_left_top_y-value_given,pixel_left_bottom[1],pixel_right_top[1],pixel_right_bottom[1]]


dot_gap_x = (screen_x[3]-screen_x[1])/8
dot_gap_y = (screen_y[2]-screen_y[3])/8
dot_scatter_x = []
dot_scatter_y = []
baseline_x_temp = []
baseline_y_temp = []

screen_bottom_line_y_segment = np.linspace(screen_y[1],screen_y[3],9)

for i in range(9):
    for j in range(8,-1,-2):
        dot_scatter_x.append(dot_gap_x*i+screen_x[1])
        dot_scatter_y.append(dot_gap_y * abs(j-8)+ screen_bottom_line_y_segment[i])
        plt.scatter(dot_scatter_x,dot_scatter_y, color='black')
        baseline_x_temp.append(i)
        baseline_y_temp.append(abs(j-8))
        # print(j)
print(dot_scatter_y)
print(baseline_y_temp)
# plt.show()
# print(dot_scatter_y)
# print(baseline_x)
# change order
order = [4,9,14,19,24,29,34,39,44]
order2 = []
for i in range(5):
    for j in order:
        order2.append((j-i))

# in real dimension cm
screen_width_cm = 154
screen_height_cm = 87
screen_width_pixel = x[2]-x[0]
screen_height_pixel = y[2]-y[3]
pixel_per_cm = 154/screen_width_pixel
with open('pixel_per_cm.txt','w') as f:
    f.write(str(pixel_per_cm))
dot_cm_x = []
dot_cm_y = []

for i in range(len(dot_scatter_x)):
    # dot_cm_x.append((dot_scatter_x[i]+x[1])*pixel_per_cm)
    # dot_cm_y.append((dot_scatter_y[i] + y[1]) * pixel_per_cm)
    dot_cm_x.append((dot_scatter_x[i]) * pixel_per_cm)
    dot_cm_y.append((dot_scatter_y[i]) * pixel_per_cm)
print(dot_cm_y)
# #change order
dot_cm_x = [dot_cm_x[i] for i in order2]
dot_cm_y = [dot_cm_y[i] for i in order2]
baseline_x_temp = [baseline_x_temp[i] for i in order2]
baseline_y_temp = [baseline_y_temp[i] for i in order2]
print(baseline_y_temp)
print(baseline_x_temp)

# Get world camera scene dimensions in cm
world_x_range_pixel = world_x_max-world_x_min
world_y_range_pixel = world_y_max-world_y_min
world_x_range_cm = world_x_range_pixel*pixel_per_cm
world_y_range_cm = world_y_range_pixel*pixel_per_cm

f = open('world_cm.txt','w')
f.write(str(world_x_range_cm)+"\n")
f.write(str(world_y_range_cm))
f.close()

# change coordinate: origin from left-top to left_bottom
# dot_cm_y = [(world_y_range_cm-dot_cm_y[i]) for i in range(len(dot_cm_y))]

dot_cm_x = np.tile(dot_cm_x,3)
dot_cm_y = np.tile(dot_cm_y,3)
baseline_x_temp = np.tile(baseline_x_temp,3)
baseline_y_temp = np.tile(baseline_y_temp,3)
# print(baseline_y_temp)

df_cm = pd.DataFrame(dot_cm_x,columns=['dot_x_cm'])
df_cm['dot_y_cm'] = pd.DataFrame(dot_cm_y)
df_cm['baseline_x_temp'] = pd.DataFrame(baseline_x_temp)
df_cm['baseline_y_temp'] = pd.DataFrame(baseline_y_temp)

df_dot = pd.read_csv('dot_data_0_8_8_8_2022_12_13_18_12_27.csv')
df_cm = pd.concat([df_dot,df_cm],axis=1)
df_cm.to_csv('cm.csv',index=False)

# Get world camera scene dimensions in cm
world_x_range_pixel = world_x_max-world_x_min
world_y_range_pixel = world_y_max-world_y_min
world_x_range_cm = world_x_range_pixel*pixel_per_cm
world_y_range_cm = world_y_range_pixel*pixel_per_cm
f = open('world_cm.txt','w')
f.write(str(world_x_range_cm)+"\n")
f.write(str(world_y_range_cm))
f.close()