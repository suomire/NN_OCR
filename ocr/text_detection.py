import numpy as np
import pytesseract
import cv2
import os

NW_WIDTH = 10
NW_HEIGHT = 10
image = cv2.imread("..\example_01.png")
orig = image.copy()
print(image.shape)
(orig_h, orig_w) = image.shape[:2]
(newW, newH) = (NW_WIDTH, NW_HEIGHT)
# change ratio
rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]
# 1- output probabilities, 2- bounding box coord
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
path = ""
net = cv2.dnn.readNet(path)
