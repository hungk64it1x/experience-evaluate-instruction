from pickletools import optimize
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
from PIL import Image


rtsp_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
video_stream = VideoStream(rtsp_url).start()
prev_time = 0
new_time = 0


while True:
    frame = video_stream.read()
    if frame is None:
        continue
    # cv2.imwrite('test.jpg', frame)
    frame = imutils.resize(frame, width=1024)
    img_pil = Image.fromarray(frame)
    img_pil.save('1.jpg', optimize=True, quality=5)
    image = cv2.imread('1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('AsimCodeCam', image)
    new_time = time.time()
    fps = int(1/ (new_time - prev_time))
    prev_time = new_time
    # cv2.putText(frame, f'FPS: {fps}', (0, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 2)
    cv2.putText(image, f'FPS: {fps}', (0, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 2)
    # cv2.imshow('AsimCodeCam', frame)
    cv2.imshow('CodeCam', image)

    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    

cv2.destroyAllWindows()
video_stream.stop()

