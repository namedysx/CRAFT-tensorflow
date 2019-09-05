import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import math

def Random_scale(img, bbox):
    h, w = img.shape[:2]
    scale = random.uniform(0.8, 2.0)
    bbox[0,:,:] *= scale
    bbox[1,:,:] *= scale
    img = cv2.resize(img, (h*scale, w*scale))
    return img, bbox

def Random_rot(img, bbox):
    h, w = img.shape[:2]
    angle = random.randint(0, 90)
    # scale = random.uniform(0.8, 2.0)
    rx0, ry0 = w / 2, h / 2
    bbox[0,:,:] = (bbox[0,:,:] - rx0)*math.cos(math.radians(angle)) - (bbox[1,:,:] - ry0)*math.sin(math.radians(angle)) + rx0
    bbox[1,:,:] = (bbox[0,:,:] - rx0)*math.sin(math.radians(angle)) + (bbox[1,:,:] - ry0)*math.cos(math.radians(angle)) + ry0
    # bbox[bbox<0] = 0
    # bbox[0,:,:][bbox[0,:,:]>h] = h
    # bbox[1,:,:][bbox[1,:,:]>w] = w
    matRotation = cv2.getRotationMatrix2D((rx0, ry0), (360-angle), 1.0)
    imgRotation = cv2.warpAffine(img, matRotation, (h, w), borderValue=(0, 0, 0))
    return imgRotation, bbox

def Random_filp(img, bbox):
    h, w = img.shape[:2]
    p = 0.5
    # 水平翻转
    if random.random() > p:
        img = cv2.flip(img, 1)
        bbox[0,:,:] = 2*(w/2) - bbox[0,:,:]
    elif random.random() > p:
        img = cv2.flip(img, 0)
        bbox[1,:,:] = 2*(h/2) - bbox[1,:,:]
    return img, bbox

def Random_crop(img, bbox):
    img, bbox = Random_scale(img, bbox)
    h, w = img.shape[:2]
    x_left_up = random.randint(0, int(0.2*h))
    y_left_up = random.randint(0, int(0.2*w))
    x_right_bottom = random.randint(int(0.8*h), h)
    y_right_bottom = random.randint(int(0.8*w), w)
    img = img[x_left_up:x_right_bottom, y_left_up:y_right_bottom]
    bbox[0,:,:] = bbox[0,:,:] - x_left_up
    bbox[1,:,:] = bbox[1,:,:] - y_left_up
    bbox[bbox<0] = 0
    bbox[0,:,:][bbox[0,:,:]>x_right_bottom] = x_right_bottom
    bbox[1,:,:][bbox[1,:,:]>y_right_bottom] = y_right_bottom
    return img, bbox
