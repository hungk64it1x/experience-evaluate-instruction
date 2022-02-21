"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import numpy as np
import os
from PIL import Image
import cv2
from scipy.spatial import distance as dst
import torch
from color import *

from flask import Flask, render_template, request, redirect

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
    
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=768, augment=False)
        preds   = results.pandas().xyxy[0]
        if len(preds):
            bboxes  = preds[['xmin','ymin','xmax','ymax']].values
            confs   = preds.confidence.values
        coords = []
        areas = []
        for index, (box, conf) in enumerate(zip(bboxes, confs)):
            img = np.array(img)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            car_crop = img[y1: y2, x1: x2]
            hsv_img = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
            for c in range(len(list_count_color)):
                list_count_color[c] = color_detection(count_min_color[c], count_max_color[c], hsv_img, car_crop)
            max_value = max(list_count_color)
            name_color = list_name_colors[list_count_color.index(max_value)]
            coords.append([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])
            areas.append((x2 - x1) * (y2 - y1))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(coords)> 1:
            coord1, coord2 = coords[0], coords[1]
            area1, area2 = areas[0], areas[1]
            distance = dst.euclidean(coord1, coord2)
            cv2.line(img, tuple(coord1), tuple(coord2), (0, 255, 0), 2)
            temp = int(distance / 2)
            ds = '%.2f'%(distance)
            cv2.putText(img, str(ds), (coord1[0] + temp, coord1[1]), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        else:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")

        return redirect("static/image0.jpg")
    
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(r'yolov5', 
                       'custom', 
                       path=r'yolov5x6.pt',
                       source='local',
                       force_reload=True)  # local repo
    model.iou = 0.45
    model.conf = 0.5
    model.classes = 2
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
