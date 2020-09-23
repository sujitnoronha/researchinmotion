import cv2
import numpy as np

width = 500
height = 500

blank_image = np.zeros(
    (width, height,3)
)

blank_image[:] = (0,0,0)
print(blank_image)

cv2.imshow('img',blank_image)
cv2.waitKey(0)