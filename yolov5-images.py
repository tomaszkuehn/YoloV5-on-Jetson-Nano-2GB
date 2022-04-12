# IP camera image capture (player) OpenCV - all in memory
# Apache 2.0 license
# Copyright (C) 2022 Tomasz Kuehn v0.3

import cv2
import torch
from PIL import Image
import time
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
from threading import Thread


def thread_image():
    global r
    #r = requests.get("http://192.168.0.250/cgi-bin/jpg/image.cgi", stream = False, auth = HTTPBasicAuth("Admin", "1234"))
    #r = requests.get("http://192.168.88.209:8080/shot.jpg", stream = False) 
    r = requests.get("http://192.168.88.176:8080/shot.jpg", stream = False) 


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained = 'True', device='0') #device 0 is Jetson cuda device

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

#model.cuda()
fpsstart = 0
color = (255, 255, 255)

image = np.zeros(shape=[360, 640, 3], dtype=np.uint8)
cv2.imshow("Image", image)
cv2.waitKey(2000)
t = Thread(target = thread_image)
t.start()
while 1:
    #tt = time.time()
    t.join()
    r1 = r
    t = Thread(target = thread_image)
    t.start()
    try:
        image = cv2.imdecode(np.frombuffer(r1.content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR) 
        #points = (320, 240)
        #image = cv2.resize(image, points, interpolation= cv2.INTER_LINEAR)
        #diff = time.time() - tt
        #print("[INFO] IMAGE took {:.6f} seconds".format(diff))

        tt = time.time()
        # Inference
        results = model(image, size=320)  # includes NMS
        diff = time.time() - tt
        print("[INFO] YOLO took {:.6f} seconds".format(diff))
    
        # Results
        #results.print()  
        results.render() #render boxes around objects
        diff = time.time() - fpsstart
        fpsstart = time.time()
        fps = 1.0 / diff
        #print("FPS: %.2f" % fps)
        
        cv2.putText(results.imgs[0], 'FPS ' + '%.2f' % fps, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 3) #cv2.LINE_AA
        cv2.putText(results.imgs[0], 'FPS ' + '%.2f' % fps, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,0), 2)
        cv2.imshow("Image", results.imgs[0])
        cv2.waitKey(1)
    except:
        print("No image")

#results.xyxy[0]  # im1 predictions (tensor)
#results.pandas().xyxy[0]  # im1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

