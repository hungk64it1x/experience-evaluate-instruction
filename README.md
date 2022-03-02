# Expertise evaluation of Image processing and computer vision skill

--------------------------------------------
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
### Task 2: Calculating distance between any cars
### Method:
![image](https://user-images.githubusercontent.com/80585483/154884630-a175c36f-2b5a-4c47-966b-63cad6d9869b.png)
#### 1. Using yolov5 (yolov5s, yolov5x, ...) to detect car (pretrain weight with COCO dataset contains cars so we don't need to retrain with any dataset)
##### Get weights in repo: [yolov5](https://github.com/ultralytics/yolov5/releases) and put into task2 folder.
#### 2. Using euclide distance in sklearn to compute distance between 2 cars in image.
#### How to run:
#### Install needed packages:
```
cd task2
pip install -r requirements.txt
```
#### Run:
```
python webapp.py --port 5003
```
------------------------------------------
### Task 3: Korea license plate detection
### Method:
![image](https://user-images.githubusercontent.com/80585483/154885918-fec1d7e9-0118-4988-bd0a-700dcd54b1d8.png)
#### 1. Using WPOD-Net to detect license plate with cars in image
##### Using wpod-net instead of yolov5 (etc) because yolov5 can not crop close license plate like:
![image](https://user-images.githubusercontent.com/80585483/154886159-82905baf-e75a-4cdc-9801-4e3cd890abaa.png)
But in wpod-net it can crop very close.
![image](https://user-images.githubusercontent.com/80585483/154886701-1bf712fd-d967-473a-901b-4024eb9dbade.png)
#### 2. Deep text recognize
After that we have a crop of LP we feed it into an model of deep text recognition, in this task i use TPS-Resnet-BiLSTM-CTC to find out target characters.
#### How to run:
#### Download pretrained weight for wpod and CTC at: [weight](https://drive.google.com/drive/folders/1l0mSQLP9x-ujICBHYpxVpo6F9JCTZBNm?usp=sharing) then put it into folder weights in task3
#### Install needed packages:
```
cd task3
pip install -r requirements.txt
```
#### Run:
```
python webapp.py --port 5004
```
--------------------------------------
### Task 4: RTSP compression
### Method:
![image](https://user-images.githubusercontent.com/80585483/155496099-655481b3-6b6c-47e2-896a-829f910b5828.png)
### task 4 ver 1: Use save image with PIL
#### Step 1: Run RTSP video realtime with python flask
#### Step 2: With each frame I download it to my local and compress by PIL library.
#### Step 3: Upload result image back to flask realtime
### task 4 ver 2: Use this repo to compress image (frame) [url](https://github.com/abhinav-TB/JPEG-IMAGE-COMPRESSION)
#### Step 1: Compress video using DCT (Discrete Cosine Transformation)
#### Step 2: Run RTSP after compress (Realtime)

#### How to run:
#### Install needed packages:
```
pip install -r  requirements.txt
cd task4_v1 (or task4_v2)
```
#### Run:
```
python run.py --port 5000
```
--------------------------------------
### References:
- [yolov5 repo](https://github.com/ultralytics/yolov5)

- [detectron2](https://github.com/facebookresearch/detectron2)

- [deep text recognition](https://github.com/clovaai/deep-text-recognition-benchmarkhttps://github.com/clovaai/deep-text-recognition-benchmark)
- [wpod net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)
- [image compression](https://github.com/abhinav-TB/JPEG-IMAGE-COMPRESSION)
---------------------------------------
#### If you have any questions feel free to contact me 😄
