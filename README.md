# YoloV5-on-Jetson-Nano-2GB
YoloV5 on Jetson Nano 2GB or 4GB

![Screenshot from 2022-04-09 14-35-01](https://user-images.githubusercontent.com/30973162/162574451-938f769a-368f-4457-87a6-77d5a7d439be.png)

Over 10 fps with small model. Python script yolov5-images.py to process images from IP camera.

First free some memory - use Xfce desktop manager, uninstall teamviewer. You should have 0.6-0.7GB memory consumed running the jtop only. And you need to increase swap file to 3GB. The easiest way using jtop (preinstalled).

The variant with http auth is for a security camera. The other for an Android app: https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US&gl=US

