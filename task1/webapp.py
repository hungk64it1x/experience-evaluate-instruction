"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
from distutils.log import debug
import io
import numpy as np
import os
from PIL import Image
import cv2
from scipy.spatial import distance as dst
import torch
from detect import *
from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    detector = Detector()
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        img = np.array(pil_img) 
        img = img[:, :, ::-1].copy() 
        img = detector.onImage(image=img)
        img.save("static\image0.jpg", format="JPEG")
        return redirect("static\image0.jpg")
    
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5003, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat
