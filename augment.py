import os
import numpy as np
import cv2
import random

def crop_img(src,top_left_x,top_left_y,crop_w,crop_h):
    '''裁剪图像
    Args:
        src: 源图像
        top_left,top_right:裁剪图像左上角坐标
        crop_w,crop_h：裁剪图像宽高
    return：
        crop_img:裁剪后的图像
        None:裁剪尺寸错误
    '''
    rows, cols = src.shape[0: 2]
    row_min,col_min = int(top_left_y), int(top_left_x)
    row_max,col_max = int(row_min + crop_h), int(col_min + crop_w)
    if row_max > rows or col_max > cols:
        print("crop size err: src->%dx%d,crop->top_left(%d,%d) %dx%d"%(cols, rows, col_min, row_min,int(crop_w),int(crop_h)))
        return None
    crop_img = src[row_min:row_max, col_min:col_max]
    return crop_img

def crop_imgs(img, label, crop_type='RANDOM_CROP',crop_n=1, dsize=(0, 0), random_wh=False):
    '''
    Args：
        imgs_dir: 待放缩图片
        crop_type:裁剪风格 ['RANDOM_CROP','CENTER_CROP','FIVE_CROP']
        crop_n: 每原图生成裁剪图个数
        dsize:指定crop宽高（w,h），与random_wh==True互斥生效
        random_wh：随机选定裁剪宽高
    '''
    imgh, imgw = img.shape[0: 2]
    # fw, fh: 当random_wh == False时为crop比例，否则为随机crop的宽高比例下限

    fw = random.uniform(0.2, 0.98)
    fh = random.uniform(0.2, 0.98)
    crop_imgw, crop_imgh = dsize
    if dsize == (0, 0) and not random_wh:
        crop_imgw = int(imgw * fw)
        crop_imgh = int(imgh * fh)
    elif random_wh:
        crop_imgw = int(imgw * (fw + random.random() * (1 - fw)))
        crop_imgh = int(imgh * (fh + random.random() * (1 - fh)))

    if crop_type == 'RANDOM_CROP':
        crop_top_left_x, crop_top_left_y = random.randint(0, imgw - crop_imgw - 1), random.randint(0, imgh - crop_imgh - 1)
    elif crop_type == 'CENTER_CROP':
        crop_top_left_x, crop_top_left_y = int(imgw / 2 - crop_imgw / 2), int(imgh / 2 - crop_imgh / 2)
    elif crop_type == 'FIVE_CROP':
        crop_top_left_x, crop_top_left_y = 0, 0
    else:
        print('crop type wrong! expect [RANDOM_CROP,CENTER_CROP,FIVE_CROP]')

    croped_img = crop_img(img, crop_top_left_x, crop_top_left_y, crop_imgw, crop_imgh)
    croped_label = crop_img(label, crop_top_left_x, crop_top_left_y, crop_imgw, crop_imgh)

    #丢弃正样本较少的
    tmp = croped_label.copy()
    tmp[tmp > 0] = 1
    if np.sum(tmp) < 500:
        return img, label
    else:
        return croped_img, croped_label

def rot_img_and_padding(img, rot_angle, scale=1.0):
    '''
    以图片中心为原点旋转
    Args:
        img:待旋转图片
        rot_angle:旋转角度，逆时针
        scale:放缩尺度
    return:
        imgRotation:旋转后的cv图片
    '''
    img_rows, img_cols = img.shape[:2]
    cterxy = [img_cols//2, img_rows//2]

    matRotation = cv2.getRotationMatrix2D((cterxy[0], cterxy[1]), rot_angle, scale)
    imgRotation = cv2.warpAffine(img, matRotation, (img_cols, img_rows))
    return imgRotation

def rand_rot(img, label):
    '''
    :param img: [H, W, 3]
    :param lable: [H, W, 2]
    :return:
    '''
    angle = random.randint(0, 180)
    scale = random.uniform(0.9, 1.5)
    res_img = rot_img_and_padding(img, angle, scale)
    res_label = rot_img_and_padding(label, angle, scale)
    return res_img, res_label

def rand_flip(img, label):
    '''图片翻转'''
    flag = random.random()

    if flag < 0.3333:
        res_img = cv2.flip(img, 1)
        res_label = cv2.flip(label, 1)
    elif (flag >= 0.3333) and (flag < 0.6666):
        res_img = cv2.flip(img, -1)
        res_label = cv2.flip(label, -1)
    else:
        res_img = cv2.flip(img, 0)
        res_label = cv2.flip(label, 0)

    return res_img, res_label

def random_color_distort(img, label, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    在图片的HSV空间进行扭曲，还有亮度调整
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)#限幅
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)#转换色彩空间

    return img, label

def tranc(img, label):
    img = cv2.transpose(img)
    label = cv2.transpose(label)
    return img, label

def rand_augment(img, label):
    '''随机选择一种数据增强'''
    # flag = random.random()
    # print(flag)
    if random.random() < 0.5:
        # 随机裁剪
        res_img, res_label = crop_imgs(img, label)
        if random.random() < 0.5:
            res_img, res_label = tranc(res_img, res_label)
    elif random.random() < 0.5:
        # 随机翻转
        res_img, res_label = rand_flip(img, label)
        if random.random() < 0.5:
            res_img, res_label = tranc(res_img, res_label)
    # elif (flag >= 0.5) and (flag < 0.75):
    #     # 随机旋转
    #     res_img, res_label = rand_rot(img, label)
    elif random.random() < 0.5:
        # 随机色度变换
        res_img, res_label = random_color_distort(img, label)
        if random.random() < 0.5:
            res_img, res_label = tranc(res_img, res_label)
    else:
        res_img, res_label = img, label
    return res_img, res_label

if __name__ == '__main__':
    img = cv2.imread('./textimg/image.png')
    label = cv2.imread('./textimg/weight.png')
    label = label[:, :, 0:2]
    res_i, res_l = rand_augment(img, label)
    cv2.imshow('s', res_i)
    cv2.waitKey()