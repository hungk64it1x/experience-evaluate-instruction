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