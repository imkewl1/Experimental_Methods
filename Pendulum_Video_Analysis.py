import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

video_path = "C:/Users/jerem/python/WIP/IMG_2563.mov"


cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# print(fps)

# # These are good for the large angle video
# x = 700
# y1 = 600
# y2 = 800
# These are good for the small angle video
x1 = 200
x2 = 2000
y1 = 150
y2 = 300
myargmax = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_array = np.array(frame)
    height, width, channels = img_array.shape

    # meanblue.append(np.mean(img_array[y1:y2,x,0]))
    # meangreen.append(np.mean(img_array[y1:y2,x,1]))
    # meanred.append(np.mean(img_array[y1:y2,x,2]))
    # mycolor.append( np.mean([meanred[-1], meangreen[-1]]) )
    
    if 'frame_last' in locals():
        diff_array = np.array(cv2.subtract(frame,frame_last) )
        sub_array = diff_array[y1:y2,x1:x2]
        height, width, channels = sub_array.shape
        sumys = []
        for x in range(width):
            sumy = 0
            for y in range(height):
                sumy += sub_array[y,x,0]
            sumys.append( sumy )
        myargmax.append(np.argmax(np.array(sumys)))
                
        # print("Frame shape:", frame.shape)  # e.g. (height, width, 3)
        # height, width, channels = frame.shape
        # print("Trying to slice y1:y2 =", y1, y2, "x1:x2 =", x1, x2)

        # if y2 > height or x2 > width:
        #     print("WARNING: ROI is out of bounds!")
        #     # Possibly break, or adjust y2, x2 accordingly
        #cv2.imshow('Processed Frame', img_array[y1:y2,x1:x2,0])
        cv2.imshow('Processed Frame', diff_array[y1:y2,x1:x2,0])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

    frame_last = frame
    
    
cap.release()
cv2.destroyAllWindows()


myargmax_arr = np.array(myargmax)

times = np.arange(len(myargmax_arr)) / fps

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

d_amplitude = np.gradient(myargmax_arr, times[1])

# smooth
for i in range(1, len(myargmax_arr)-1):
    if (np.abs(myargmax_arr[i-1]-myargmax_arr[i]) > 30) and (np.abs(myargmax_arr[i+1]-myargmax_arr[i]) > 30):
        myargmax_arr[i] = (myargmax_arr[i-1] + myargmax_arr[i+1])/2

T = np.diff[times[np.diff(np.sign(d_amplitude) == 2)]]


# extract half periods

my_t_arglist = []
for t in np.arange(0, 37, T):
    my_t_arglist.append(np.argmax(myargmax_arr[(times>t) & (times<t+T)]) + len(myargmax_arr[times<=t]))
    my_t_arglist.append(np.argmin(myargmax_arr[(times>t) & (times<t+T)]) + len(myargmax_arr[times<=t]))


my_amplitudes = np.abs(myargmax_arr[np.array(my_t_arglist)])
my_amplitudes_t = times[np.array(my_t_arglist)]

mean = np.mean(my_amplitudes)
my_amplitudes_corrected = np.abs(my_amplitudes - mean)

ax1.plot(times, myargmax_arr,'.k')
ax1.set_xlabel('time [s]')
ax1.set_ylabel('pixel')
ax1.set_xlim([0,45])

ax2.plot(my_amplitudes_t, my_amplitudes_corrected,'.k')
ax2.set_xlabel('time [s]')
ax2.set_ylabel('pixel')

append_to_csv("C:/Users/jerem/python/WIP/Data11.csv", times, myargmax_arr, "Trial 1")
append_to_csv("C:/Users/jerem/python/WIP/Data22.csv", my_amplitudes_t, my_amplitudes_corrected, "Trial 1")

plt.show()
