import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from utils import AttnLabelConverter
from model import Model
from dataset import NormalizePAD
from PIL import Image,ImageDraw,ImageFont
import math
import os
from time import time
from implements import *

global thresh
thresh = 0.5

def ConvertToTensor(s_size, src):
    imgH = s_size[0]
    imgW = s_size[1]
    input_channel = 3 if src.mode == 'RGB' else 1
    transform     = NormalizePAD((input_channel, imgH, imgW))
    w, h          = src.size
    ratio         = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)
    resized_image = src.resize((resized_w, imgH), Image.BICUBIC)
    tmp           = transform(resized_image)
    img_tensor    = torch.cat([tmp.unsqueeze(0)], 0)
    img_tensor    = rgb_to_grayscale(img_tensor)
    return img_tensor

# OCR Recognition
def Recognition(opt,img):
    # static w,h
    s_size = [opt.imgH, opt.imgW]
    text   = []

    if img:
        src               = ConvertToTensor(s_size, img)

        batch_size        = src.size(0)
        image             = src.to(device)

        length_for_pred   = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred     = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index    = preds.max(2)
        preds_str         = converter.decode(preds_index, length_for_pred)

        preds_prob        = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        pred_EOS          = preds_str[0].find('[s]')
        pred              = preds_str[0][:pred_EOS]
        pred_max_prob     = preds_max_prob[0][:pred_EOS]
        confidence_score  = pred_max_prob.cumprod(dim=0)[-1]

        if confidence_score >= thresh:
            text.append(pred)
        else:
            text.append('Missing OCR')
        
        text.append(confidence_score) 

    else:
        text.append('Missing Plate')

    return text

def GetCoordinate(cor):
    pts      = []
    x_coor   = cor[0][0]
    y_coor   = cor[0][1]

    for i in range(4):
        pts.append([int(x_coor[i]),int(y_coor[i])])

    pts      = np.array(pts, np.int32)
    pts      = pts.reshape((-1,1,2))
    return pts

if __name__=='__main__':
    global device,converter
    home                = os.getcwd()
    wpod_net_path       = 'task3\weights\wpod-net.json'
    load_model(wpod_net_path)
    device              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt                 = CreateParser()
    cudnn.benchmark     = True
    cudnn.deterministic = True
    opt.num_gpu         = torch.cuda.device_count()
    converter           = AttnLabelConverter(opt.character)
    opt.num_class       = len(converter.character)
    if opt.rgb:
        opt.input_channel = 3

    model    = Model(opt)
    model    = torch.nn.DataParallel(model).to(device)

    # load model
    model.load_state_dict(torch.load(r'task3\weights\v1.6-best_accuracy.pth', map_location=device))
    idx      = 0
    dir_path = r'D:\VSCode\CarDetection\license-plate'
    
    img_list = os.listdir(dir_path)

    # predict
    model.eval()           
    with torch.no_grad():  
        while True:
            t = time()
            img_path = os.path.join(dir_path, img_list[idx])
            dst = preprocess_image(img_path)
            
            img, cor = Detect_plate(dst)
            print('Detect Number Plate Time : ', time() - t)
            t1 = time()

            if img:
                img  = img.resize((435,100))
            result   = Recognition(opt,img)
            # plt.imshow(img)
            # plt.show()
            # print('OCR Recognition Time : ', time() - t1)
            # print('Total Process Time : ',time() - t)

            # cv2 image to PIL image, Required Draw text
            dst      = Image.fromarray((dst * 255).astype(np.uint8))
            # pillow font & draw Object
            font     = ImageFont.truetype(r'task3\fonts\gothic.ttf',size=30)
            draw     = ImageDraw.Draw(dst)
            
            # draw ocr
            # draw.text((30,30),result[0],(0,255,0),font=font) # predict text or Missing str
            dst      = np.array(dst)
            dst      = cv2.cvtColor(dst,cv2.COLOR_RGB2BGR)
            if img:
                img  = np.array(img)
                img  = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            if cor:
                pts  = GetCoordinate(cor)
                cv2.polylines(dst,[pts],True,color=(0,255,0),thickness=1)
                del pts
            cv2.imshow('car image', dst)

            del cor, draw, font, img
            if cv2.waitKey()   == ord('n'):
                if idx         == len(img_list):
                    pass
                idx += 1
            elif cv2.waitKey() == ord('p'):
                if idx         == 0:
                    pass
                idx -= 1
            elif cv2.waitKey() == 27:
                break