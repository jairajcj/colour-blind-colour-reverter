import cv2
import numpy as np

class ColorBlindnessCorrector:
    pass

# RGB to LMS matrix
RGB_TO_LMS = np.array([[0.31399022, 0.63951294, 0.04649755],[0.15537241, 0.75789446, 0.08670142],[0.01775239, 0.10944209, 0.87256922]])

LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)

