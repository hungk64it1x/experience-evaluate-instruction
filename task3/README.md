
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
