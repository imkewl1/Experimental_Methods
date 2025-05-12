                 # The way this works is:
#  the mouse is used to record the scale (two data points)
#  followed by the mouse being used to record the projectile position
#  hitting a key records the data and moves to the next option or frame
#  a dot indicates where the data is going to be recorded if a key is pressed
#  the data can be re-recorded by simply using the mouse again, having not yet pressed a key
#  on mouse down, we zoom in, on mouse up, the candidate data is recorded and displayed

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
        
# video_path = 'vids/trimmed5_sec1.mov'
video_path = "C:/Users/jerem/python/WIP/IMG_1.mov"
# video_path = 'vids/trimmed6.mov'

max_num_frames = 10000
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # innaccurate
print(fps, frame_number)

zoom_factor = 4
crop_x1 = 0
crop_y1 = 0
crop_effective_zoom_factor_x = zoom_factor
crop_effective_zoom_factor_y = zoom_factor
scale_in_meters = 1.0
click_scale_x_arr = np.zeros(2)
click_scale_y_arr = np.zeros(2)
click_x_arr = np.zeros(max_num_frames)
click_y_arr = np.zeros(max_num_frames)
def on_click_xy(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_array = np.array(frame) # want to renew, in case multiple clicks
        img = zoom_image(img_array, x, y)
        cv2.imshow('image', img)        
    if event == cv2.EVENT_LBUTTONUP:
        img_array = np.array(frame) # want to renew, in case multiple clicks
        x_translated_back = crop_x1 + int(x/crop_effective_zoom_factor_x)
        y_translated_back = crop_y1 + int(y/crop_effective_zoom_factor_y)
        cv2.circle(img_array, (x_translated_back, y_translated_back), 10, (0, 0, 255), -1)
        cv2.imshow('image', img_array)
        click_x_arr[frame_index] = x_translated_back
        click_y_arr[frame_index] = y_translated_back
        # print(x_translated_back)
def on_click_scale(event, x, y, p1, p2):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_array = np.array(frame) # want to renew, in case multiple clicks
        img = zoom_image(img_array, x, y)
        cv2.imshow('image', img)        
    if event == cv2.EVENT_LBUTTONUP:
        img_array = np.array(frame) # want to renew, in case multiple clicks
        x_translated_back = crop_x1 + int(x/crop_effective_zoom_factor_x)
        y_translated_back = crop_y1 + int(y/crop_effective_zoom_factor_y)
        cv2.circle(img_array, (x_translated_back, y_translated_back), 10, (0, 255, 0), -1)
        cv2.imshow('image', img_array)
        click_scale_x_arr[scale_index] = x_translated_back
        click_scale_y_arr[scale_index] = y_translated_back
def zoom_image(image, x, y):
    height, width = image.shape[:2]

    global crop_x1
    global crop_y1
    global crop_effective_zoom_factor_x
    global crop_effective_zoom_factor_y
    crop_x1 = x - width//(2*zoom_factor)
    crop_y1 = y - height//(2*zoom_factor)
    crop_x2 = x + width//(2*zoom_factor)
    crop_y2 = y + height//(2*zoom_factor)
    crop_effective_zoom_factor_x = zoom_factor
    crop_effective_zoom_factor_y = zoom_factor
    if crop_x1 < 0:
        crop_effective_zoom_factor_x = width / (width/zoom_factor + crop_x1)
        crop_x1 = 0
    if crop_y1 < 0:
        crop_effective_zoom_factor_y = height / (height/zoom_factor + crop_y1)
        crop_y1 = 0        
    if crop_x2 > width-1:
        crop_effective_zoom_factor_x = width / (width/zoom_factor - (crop_x2-width))
        crop_x2 = width-1
    if crop_y2 > height-1:
        crop_effective_zoom_factor_y = height / (height/zoom_factor - (crop_y2-height))
        crop_y2 = height-1        
    # print('zoom crop: ', crop_x1, crop_y1)
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]    
    resized_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

    return resized_image

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index == 0:
        # record first click for scale
        scale_index = 0
        img_array = np.array(frame)                        
        cv2.imshow('image', img_array)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_click_scale)
        cv2.waitKey(0)
        
        # record second click for scale
        scale_index = 1
        img_array = np.array(frame)                        
        cv2.imshow('image', img_array)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_click_scale)
        cv2.waitKey(0)

    # begin recording clicks for location
    img_array = np.array(frame)                        
    cv2.imshow('image', img_array)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_click_xy)
    cv2.waitKey(0)

    frame_index += 1

cap.release()
cv2.destroyAllWindows()

click_x_arr = click_x_arr[0:frame_index]
click_y_arr = click_y_arr[0:frame_index]
times_arr = np.arange(len(click_x_arr)) / fps
scale_in_meters_arr = np.array(scale_in_meters)

outfile = "trimmed5_only_peaks_L"
np.savez(outfile, click_x_arr, click_y_arr, times_arr, scale_in_meters_arr)
