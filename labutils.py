import cv2
import numpy as np

def crop(img, topbottom_leftright):
	top = topbottom_leftright[0][0]
	bottom = img.shape[0]- topbottom_leftright[0][1]
	left = topbottom_leftright[1][0]
	right = img.shape[1] - topbottom_leftright[1][1]
	return img[top:bottom, left:right]

def resize(img, new_shape):
	return cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)