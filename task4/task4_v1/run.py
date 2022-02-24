from flask import Flask,render_template,Response, request, redirect, url_for
import cv2
import imutils
from PIL import Image
from imutils.video import VideoStream
import argparse
import time

temp = 0
app=Flask(__name__)
rtsp_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
video_stream = VideoStream(rtsp_url).start()
def generate_frames(data):
    
    prev_time = 0
    new_time = 0
    while True:
        ## read the camera frame
        frame = video_stream.read()
        quality = int.from_bytes(data, 'little') - 48
        if quality <= 0:
            quality = 100
        img_pil = Image.fromarray(frame)
        img_pil.save('1.jpg', quality=quality)
        try:
            frame = cv2.imread('1.jpg')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except:
            frame = video_stream.read()
        
        new_time = time.time()
        fps = int(1/ (new_time - prev_time))
        prev_time = new_time
        cv2.putText(frame, f'FPS: {fps}', (0, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        if frame is None:
            break
        else:
            ret,buffer = cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['POST', 'GET'])
def video():
    received_data = request.data
    return Response(generate_frames(received_data), mimetype='multipart/x-mixed-replace; boundary=frame')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 