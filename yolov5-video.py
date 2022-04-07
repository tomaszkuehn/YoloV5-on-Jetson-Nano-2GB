# OpenCV with YoloV5 on video
# Apache 2.0 license
# Copyright (C) 2022 Tomasz Kuehn v0.1

import cv2
import torch
from PIL import Image
import time
import requests
from requests.auth import HTTPBasicAuth
import numpy as np

def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='0') #for cuda device



model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 1000  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

#model.cuda()
fpsstart = 0

w = 1280
h = 720
fps = 30
output_video = '/home/jetson/Videos/test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

pathIn = r'/home/jetson/Videos/myvid_Yv4-2022-04-05_18.38.36.mp4'
vidcap = cv2.VideoCapture(pathIn)

success = True
while success:
    #image = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_ANYCOLOR)
    success,image = vidcap.read()
    if success:
        points = (w, h)
        image = cv2.resize(image, points, interpolation= cv2.INTER_LINEAR)
        tt = time.time()
        # Inference
        results = model(image, size=640)  # includes NMS
        diff = time.time() - tt
        print("[INFO] YOLO took {:.6f} seconds".format(diff))
        # Results
        results.render()
        diff = time.time() - fpsstart
        fpsstart = time.time()
        fps = 1.0 / diff
        color = (255, 200, 255)
        cv2.putText(results.imgs[0], 'FPS ' + '%.2f' % fps, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)
        cv2.imshow("Image", results.imgs[0])
        cv2.waitKey(10)
        writer.write(results.imgs[0])
        #writer.write(pil_to_cv(results.imgs[0]))

writer.release() 
vidcap.release()
print("Done")



