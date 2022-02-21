import colorsys
import numpy as np
import io
import torch
import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageTk
import matplotlib.colors as mplc
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask, PolygonMasks, BitMasks
from detectron2 import model_zoo
import matplotlib.pyplot as plt
import cv2

list_name_colors = ['red', 'green', 'blue', 'white', 'black', 'silver', 'yellow', 'orange']
list_color_polygon = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (0,0,0), (0.741, 0.690, 0.694), (0, 1, 1), (0, 0.501, 1)]
list_color_rec = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255,255,255), (0,0,0), (189, 178, 177), (0, 255, 255), (0, 128, 255)]
list_color_label = [(255, 255, 255), (0, 0, 0), (255, 255, 255), (0, 0, 0), (255, 255, 255), (255, 255,255), (0,0,0), (255, 255,255)]
list_count_color = [0,0,0,0,0,0,0,0]
# count_min_color = [[170, 70, 50], [35, 52, 72], [94, 80, 2], [0, 0, 175], [5,5,5], [0,10,70], [20,100,100], [5,150,150]]
# count_max_color = [[180,255,255], [102,255,255], [126,255,255], [172,111,255], [180,255,60], [179,50,255], [40,255,255], [15, 255,255]]

count_min_color = [[159, 50, 70], [35, 52, 72], [90, 80, 2], [0, 0, 175], [8,8,8], [0,10,40], [25,90,100], [10,50,70]]
count_max_color = [[180,255,255], [89,255,255], [126,255,255], [172,111,255], [180,255,60], [180,40,200], [35,255,255], [24, 255,255]]


class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(r'COCO-InstanceSegmentation\mask_rcnn_R_50_FPN_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cpu'
        self.predictor = DefaultPredictor(self.cfg)
        self.h = None
        self.w = None
        self.scale = 1
        self.image = None

    def _convert_masks(self, masks_or_polygons, h, w):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """

        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        if isinstance(m, GenericMask):
            ret.append(m)
        else:
            ret.append(GenericMask(m, h, w))
        return ret

    def color_detection(self, min_color, max_color, hsv_img, img):
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

    def _change_color_brightness(self, color, brightness_factor):

        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color


    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
       
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)
        self._default_font_size = max(
            np.sqrt(self.h * self.w) // 90, 10 // self.scale
        )
        
        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.scale, 1),
        )
        
        return polygon
    
    def fig2img(self, fig):
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def onImage(self, image=None):
        h, w = image.shape[0], image.shape[1]
        self.image = image
        self.h = h
        self.w = w
        preds = self.predictor(image)
        viz = Visualizer(image[:,:,::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))

        output = viz.draw_instance_predictions(preds['instances'].to('cpu'))
        image_temp = output.get_image()[:,:,::-1]
        fig, ax = plt.subplots()
        for i in range(len(preds['instances'])):
            classes = preds['instances'][i].pred_classes.detach().numpy()[0]
            if classes != 2:
                continue
            temp = np.array(preds['instances'][i].pred_masks[0], dtype=np.uint8)
            box = preds['instances'][i].pred_boxes.tensor.detach().numpy()[0]
            mask_temp = np.asarray(temp)
            mask_temp = GenericMask(temp, h,w)
            mask_temp = self._convert_masks(mask_temp, h, w)
            segment = mask_temp[0].polygons
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            temp[temp > 0] = 255
            
            car_mask = np.stack([temp, temp, temp], axis=-1).reshape(h, w, 3)
            mask = np.bitwise_and(car_mask, image[:,:,::-1])
            mask = mask[y1: y2, x1: x2]
            
            mask = cv2.resize(mask, (256,256))
            hsv_img = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
            for c in range(len(list_count_color)):
                list_count_color[c] = self.color_detection(count_min_color[c], count_max_color[c], hsv_img, mask)
            
            max_value = max(list_count_color)
            p = list_count_color.index(max_value)
            name_color = list_name_colors[p]
            color = list_color_polygon[p]
            polygon = self.draw_polygon(segment[0].reshape(-1, 2), color, alpha=0.5)
            ax.add_patch(polygon)
            cv2.rectangle(image, (x1, y1), (x2, y2), list_color_rec[p], 2)
            cv2.putText(image, str(name_color), (x1, y1 + (y2 - y1)), cv2.FONT_HERSHEY_PLAIN, 2, list_color_rec[p], 3)
        ax.imshow(image[:,:,::-1])
        im = self.fig2img(fig)
        # im.show()
        # plt.show()
        return im.convert('RGB')

 

if __name__ == '__main__':
    img = cv2.imread(r'D:\VSCode\CarDetection\data\test.jpg')
    detector = Detector()
    img = detector.onImage(img)
    # img_base64 = Image.fromarray(img)
    img.show()
    # img.save("detectron2\static\image0.jpg", format="JPEG")
