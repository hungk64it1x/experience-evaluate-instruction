import argparse
import io
import os
import numpy as np
from PIL import Image
from utils import *
import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

class CFG:
    Transformation = 'TPS'
    FeatureExtraction= 'ResNet'
    SequenceModeling = 'BiLSTM'
    Prediction = 'Attn'
    num_fiducial = 20
    imgH = 32
    imgW = 100
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    batch_max_length = 25
    num_class = len(converter.character)
    thresh = 0.5

home = os.getcwd()
wpod_net_path = 'weights\wpod-net.json'
load_model(wpod_net_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(CFG)
model = torch.nn.DataParallel(model).to(device)
# load model
model.load_state_dict(torch.load(r'weights\v1.6-best_accuracy.pth', map_location=device))
model.eval()           
      
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        img = np.array(pil_img) 
        dst = process_image(img)
        img, cor = Detect_plate(dst)
        if img:
            img  = img.resize((435,100))
        result = Recognition(CFG, img, model)
        dst = Image.fromarray((dst * 255).astype(np.uint8))
        font = ImageFont.truetype(r'fonts\gothic.ttf', size=30)
        draw = ImageDraw.Draw(dst)

        draw.text((30,30),result[0],(0,255,0),font=font) 
        dst = np.array(dst)
        dst = cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
        if img:
            img = np.array(img)
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        if cor:
            pts = GetCoordinate(cor)
            cv2.polylines(dst,[pts],True,color=(0,255,0),thickness=1)

        img_base64 = Image.fromarray(dst)
        img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")

    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR")
    parser.add_argument("--port", default=5004, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)  