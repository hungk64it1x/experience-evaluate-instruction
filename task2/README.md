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

### References:
- [yolov5 repo](https://github.com/ultralytics/yolov5)

- [detectron2](https://github.com/facebookresearch/detectron2)

- [deep text recognition](https://github.com/clovaai/deep-text-recognition-benchmarkhttps://github.com/clovaai/deep-text-recognition-benchmark)
- [wpod net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)
---------------------------------------
#### If you have any questions feel free to contact me 😄
