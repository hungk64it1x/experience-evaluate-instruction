# Expertise evaluation of Image processing and computer vision skill
### Supervisor: Dr. Pham Dinh Lam
## Information:
![273964632_348781443845835_8191665654951310656_n](https://user-images.githubusercontent.com/80585483/154898566-a9bd9291-f509-4f24-a24e-f129b57136b8.jpg)


### Name: Pham Vu Hung
### School: Hanoi University of Science and Technology
### Email: hungk64it1@gmail.com
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
#### Download pretrained weight for wpod and CTC at: [weight](https://drive.google.com/drive/folders/1L7NEqBGzGzLa-OPpm4K_tHCaPx7v6HuR?usp=sharing) then put it into folder weights in task3
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
### References:
- [yolov5 repo](https://github.com/ultralytics/yolov5)

- [detectron2](https://github.com/facebookresearch/detectron2)

- [deep text recognition](https://github.com/clovaai/deep-text-recognition-benchmarkhttps://github.com/clovaai/deep-text-recognition-benchmark)
- [wpod net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)
---------------------------------------
#### If you have any questions feel free to contact me 😄
