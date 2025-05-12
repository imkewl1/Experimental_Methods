import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os
import pandas as pd

def append_to_csv(file_path, x_data, y_data, run_name):
        df_new = pd.DataFrame({
        f"{run_name}_x": x_data,
        f"{run_name}_y": y_data
     })

        if os.path.exists(file_path):
        # If the file already exists, read it
            df_existing = pd.read_csv(file_path)
        
        # To handle the possibility of different lengths, reindex both
            max_len = max(len(df_existing), len(df_new))
            df_existing = df_existing.reindex(range(max_len))
            df_new = df_new.reindex(range(max_len))
        
        # Concatenate horizontally
            df_out = pd.concat([df_existing, df_new], axis=1)
        else:
        # If no existing file, just use the new DataFrame
            df_out = df_new

    # Write it back to CSV
        df_out.to_csv(file_path, index=False)

videos = 12

for i in range(1, videos+1):
    video_path = f"C:/Users/jerem/python/WIP/IMG_{i}.mov"

    max_num_frames = 10000
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps, frame_number)

    x1 = 0
    x2 = 8000
    y1 = 660
    y2 = 715

    myargmax = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_array = np.array(frame)
        height, width, channels = img_array.shape

        if 'frame_last' in locals():
            diff_array = np.array(cv2.subtract(frame, frame_last))
            sub_array = diff_array[y1:y2, x1:x2]
            height, width, channels = sub_array.shape
            sumys = []
            for x in range(width):
                sumy = 0
                for y in range(height):
                    sumy += sub_array[y, x, 0]
                sumys.append(sumy)
            myargmax.append(np.argmax(np.array(sumys)))

            #print("frameshape:", frame.shape)
            height, width, channels = frame.shape
            #print("Trying to slice y1:y2 =", y1, y2, "x1:x2 =", x1, x2)

            #if y2 > height or x2 > width:
                #print("WSRNING: ROI is out of bounds!")
            cv2.namedWindow("win_name", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("win_name", 4000, 100)
            cv2.imshow("win_name", img_array[y1:y2, x1:x2, 0])
            #cv2.imshow('Processed Frame', img_array[y1:y2, x1:x2, 0])
            #cv2.imshow('Processed Frame', diff_array[y1:y2, x1:x2, 0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_last = frame

    seconds = int(frame_number / fps)

    cap.release()
    cv2.destroyAllWindows()

    myargmax_arr = np.array(myargmax)
    times = np.arange(len(myargmax_arr)) / fps

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    d_amplitude = np.gradient(myargmax_arr, times[1])

    # smooth
    for j in range(1, len(myargmax_arr)-1):
        if (np.abs(myargmax_arr[j-1]-myargmax_arr[j]) > 30) and (np.abs(myargmax_arr[j+1]-myargmax_arr[j]) > 30):
            myargmax_arr[j] = (myargmax_arr[j-1] + myargmax_arr[j+1])/2

    times2 = times[:-1]

    T1 = np.diff(times2[np.diff(np.sign(d_amplitude)) == 2])
    T1 = T1[(T1 > 1.55) & (T1 < 1.71)]
    T2 = np.diff(times2[np.diff(np.sign(d_amplitude)) == -2])
    T2 = T2[(T2 > 1.55) & (T2 < 1.71)]
    T = (np.mean(T1) + np.mean(T2))/2

    # extract half periods

    my_t_arglist = []


    for t in np.arange(0, seconds, T):
        my_t_arglist.append(np.argmax(myargmax_arr[(times>t) & (times<t+T)]) + len(myargmax_arr[times<=t]))
        my_t_arglist.append(np.argmin(myargmax_arr[(times>t) & (times<t+T)]) + len(myargmax_arr[times<=t]))


    my_amplitudes = np.abs(myargmax_arr[np.array(my_t_arglist)])
    my_amplitudes_t = times[np.array(my_t_arglist)]

    mean = np.mean(my_amplitudes)
    my_amplitudes_corrected = np.abs(my_amplitudes - mean)

    append_to_csv("C:/Users/jerem/python/WIP/DataA.csv", times, myargmax_arr, f"Trial_{i}")
    append_to_csv("C:/Users/jerem/python/WIP/DataB.csv", my_amplitudes_t, my_amplitudes_corrected, f"Trial_{i}")
    append_to_csv("C:/Users/jerem/python/WIP/DataC.csv", T1, T2, f"Trial_{i}")
    append_to_csv("C:/Users/jerem/python/WIP/DataD.csv", times, d_amplitude, f"Trial_{i}")




