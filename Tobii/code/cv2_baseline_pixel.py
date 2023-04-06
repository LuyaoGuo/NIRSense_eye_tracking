import cv2


# function to display the coordinates of
# of the points clicked on the image
import pandas as pd


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ',', y, ',', i)
        # print(type(i))
        df = pd.DataFrame([x], columns=['pixel_x'])
        df['pixel_y'] = pd.DataFrame([y])
        df['baseline_x'] = pd.DataFrame([(i.split('.')[0]).split('_')[0]])
        df['baseline_y'] = pd.DataFrame([(i.split('.')[0]).split('_')[1]])
        df.to_csv('pixel_coordinates.csv',mode='a', header=False, index=False)
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        filename = 'opencv' + str(i)
        cv2.imwrite(r'C:\Users\lguo8\PycharmProjects\EyeTracking_baseline\Tobii\luyao_12_13_3_light_on\20221213T071219Z\cv2_pixels\pixeled\{}'.format(filename), img)  # .png !
        cv2.imshow('image', img)
        # cv2.imwrite(r'C:\Users\lguo8\PycharmProjects\EyeTracking_baseline\Tobii\luyao_12_13_3_light_on\20221213T071219Z\cv2_pixels\pixeled', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        cv2.imshow('image', img)

import glob
image_list = (glob.glob('*.png'))
img_list = []
for i in image_list:
    try:
        if type(float(i[0]))==float:
            img_list.append(i)
    except: pass

# driver function
if __name__ == "__main__":
    # df = pd.DataFrame(['pixel_x', 'pixel_y', 'baseline_x', 'baseline_y'])
    # df.to_csv('pixel_coordinates.csv',index=False)
    # reading the image
    for i in img_list:
        img = cv2.imread(i, 1)
        img = cv2.resize(img, (960, 540))
        # displaying the image
        cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()


# # import the required library
# import cv2
#
#
# # define a function to display the coordinates of
#
# # of the points clicked on the image
# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f'({x},{y})')
#
#         # put coordinates as text on the image
#         cv2.putText(img, f'({x},{y})', (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # draw point on the image
#         cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
#
#
# # read the input image
# img = cv2.imread('snap0.jpg')
#
# # create a window
# cv2.namedWindow('Point Coordinates')
#
# # bind the callback function to window
# cv2.setMouseCallback('Point Coordinates', click_event)
#
# # display the image
# while True:
#     cv2.imshow('Point Coordinates', img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
# cv2.destroyAllWindows()
