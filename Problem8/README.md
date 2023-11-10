# Problem 8

See labels.zip for labeled/classified images from each part

## Part a

- see `resnet50.py` for the image classification 
- the classified images are included in the canvas submission

## Part b

- see `yolo.py` for the yolo comparison code
- classified images are in the canvas submission

## Part c

We followed this guide to get yolo running on the pi: [https://jordan-johnston271.medium.com/tutorial-running-yolov5-machine-learning-detection-on-a-raspberry-pi-4-3938add0f719]

## Part d

- see `detectron2.py` for detectron implementations.
- To run with different detectron models, update lines 26 and 29 with one of the following yaml urls
    - Faster R-CCN - COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
    - RetinaNet - COCO-Detection/retinanet_R_50_FPN_1x.yaml
    - Mask R-CNN - COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
