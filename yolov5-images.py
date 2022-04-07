# IP camera image capture (player) OpenCV - all in memory
# Apache 2.0 license
# Copyright (C) 2022 Tomasz Kuehn v0.1

import cv2
import torch
from PIL import Image
import time
import requests
from requests.auth import HTTPBasicAuth
import numpy as np


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='0') #device 0 is Jetson cuda device

model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

#model.cuda()
fpsstart = 0
while 1:
    r = requests.get("http://192.168.0.250/cgi-bin/jpg/image.cgi", stream = False, auth = HTTPBasicAuth("Admin", "1234"))
    #r = requests.get("http://192.168.88.209:8080/shot.jpg", stream = False)
    image = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    tt = time.time()
    # Inference
    results = model(image, size=640)  # includes NMS
    diff = time.time() - tt
    print("[INFO] YOLO took {:.6f} seconds".format(diff))
    # Results
    #results.print()  
    results.render() #render boxes around objects
    diff = time.time() - fpsstart
    fpsstart = time.time()
    fps = 1.0 / diff
    color = (255, 200, 255)
    cv2.putText(results.imgs[0], 'FPS ' + '%.2f' % fps, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
    cv2.imshow("Image", results.imgs[0])
    cv2.waitKey(1)

#results.xyxy[0]  # im1 predictions (tensor)
#results.pandas().xyxy[0]  # im1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

