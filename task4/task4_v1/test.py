from flask import Flask,render_template,Response
import cv2
import imutils
from imutils.video import VideoStream
import argparse

rtsp_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
video_stream = VideoStream(rtsp_url).start()


app=Flask(__name__)

def generate_frames():
    while True:
            
        ## read the camera frame
        frame = video_stream.read()
        if frame is None:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template(r'index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCTV")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 