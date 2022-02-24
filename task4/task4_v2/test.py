import cv2
import numpy as np
from utils import *
from app import *

cap = cv2.VideoCapture(0)
quant_size = 10
block_size = 8
while True:
    ret, frame = cap.read()
    Ycr = rgb2ycbcr(frame)
    obj=jpeg(Ycr,[5])
    quants = [quant_size]
    blocks = [(block_size,block_size)]  
    for qscale in quants:
        for bx, by in blocks:
            result = obj.intiate(qscale,bx,by)
    result = result.astype(np.uint8)

    if not ret:
        break
    cv2.imshow('ss', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
