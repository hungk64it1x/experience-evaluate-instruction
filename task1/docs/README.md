### Task 1: Cars color detection
### Method:
![image](https://user-images.githubusercontent.com/80585483/154883742-90d76221-ea9b-407d-bf5e-0c6772143c8d.png)
#### 1. Using Mask-RCNN Resnet 50 by detectron2 to segment cars instance in image 
#### 2. Classify color by get HSV color range with car mask.

#### How to run:
#### Install needed packages:
```
cd task1
pip install -e .
```
#### Run:
```
python webapp.py --port 5000
```
----------------------------------------

### References:
- [yolov5 repo](https://github.com/ultralytics/yolov5)

- [detectron2](https://github.com/facebookresearch/detectron2)

- [deep text recognition](https://github.com/clovaai/deep-text-recognition-benchmarkhttps://github.com/clovaai/deep-text-recognition-benchmark)
- [wpod net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)
---------------------------------------
#### If you have any questions feel free to contact me 😄
