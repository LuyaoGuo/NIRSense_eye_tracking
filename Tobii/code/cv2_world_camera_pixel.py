import cv2


# function to display the coordinates of
# of the points clicked on the image
import pandas as pd


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        # In the image of world camera, pixel dimensions are displayed from top-left, bottom-left, bottom_right, top_right
        print(x, ',', y, ',', i)
        # print(type(i))
        df = pd.DataFrame([x], columns=['pixel_x'])
        df['pixel_y'] = pd.DataFrame([y])
        df.to_csv('pixel_world_camera.csv',mode='a', header=False, index=False)
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 255, 0), 2)
        cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
        filename = 'opencv' + str(i)
        cv2.imshow('image', img)
        

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
    img = cv2.imread('world_camera.png', 1)
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

