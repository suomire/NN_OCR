import numpy as np
import pytesseract
import cv2
import os

image = cv2.imread("..\example_01.png")
orig = image.copy()
print(image.shape)
(orig_h, orig_w) = image.shape[:2]
