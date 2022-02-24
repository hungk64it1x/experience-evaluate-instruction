from flask import Flask,render_template,Response, request, redirect, url_for
import cv2
import imutils
import time
from PIL import Image
from imutils.video import VideoStream
import argparse
from utils import *
from app import *




temp = 0
app=Flask(__name__)

def generate_frames(data, link):
    # rtsp_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
    rtsp_url = link
    video_stream = VideoStream(rtsp_url).start()
    quant_size = 1
    block_size = 8
    try:
        quant_size = int(data)
    except:
        quant_size = 1
    prev_time = 0
    new_time = 0
    while True:
        ## read the camera frame
        frame = video_stream.read()
        Ycr = rgb2ycbcr(frame)
        obj=jpeg(Ycr,[5])
        quants = [quant_size]
        blocks = [(block_size,block_size)]
        new_time = time.time()
        fps = int(1/ (new_time - prev_time))
        prev_time = new_time  
        for qscale in quants:
            for bx, by in blocks:
                frame = obj.intiate(qscale,bx,by)
        frame = frame.astype(np.uint8)
        cv2.putText(frame, f'FPS: {fps}', (0, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)
        if frame is None:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
    data = request.form['text']
    link = request.form['link']
    return Response(generate_frames(data, link),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 