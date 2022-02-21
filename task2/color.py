import cv2
import numpy as np

list_name_colors = ['red', 'green', 'blue', 'white', 'black', 'gray', 'yellow', 'orange']
list_color_rec = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255), (0, 0, 0), (189, 176, 177), (0, 255, 255), (0, 128, 255)]
list_color_label = [(255, 255, 255), (0, 0, 0), (255, 255, 255), (0, 0, 0), (255, 255, 255), (255, 255,255), (0,0,0), (255, 255,255)]
list_count_color = [0,0,0,0,0,0,0,0]
count_min_color = [[170, 70, 50], [35, 52, 72], [94, 80, 2], [0, 0, 175], [0,0,0], [0,10,70], [20,100,100], [5,150,150]]
count_max_color = [[180,255,255], [102,255,255], [126,255,255], [172,111,255], [180,255,60], [179,50,255], [40,255,255], [15, 255,255]]

def color_detection(min_color, max_color, hsv_img, img):
    low_color = np.array(min_color, np.uint8)
    high_color = np.array(max_color, np.uint8)
    color_mask = cv2.inRange(hsv_img, low_color, high_color)
    
    height, width, channels = img.shape
    count_color = 0
    for h in range(height):
        for w in range(width):
            if color_mask[h][w] == 255:
                count_color += 1
    return count_color

