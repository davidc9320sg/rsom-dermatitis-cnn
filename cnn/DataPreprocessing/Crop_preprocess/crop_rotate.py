import numpy as np
from scipy.io import loadmat
# import matplotlib.pyplot as plt
import cv2 as cv
import random
from scipy import ndimage

debug = False


class crop_rotate_utils:
    def __init__(self):
        print('initiate crop_rotate_utils')

    #flip
    @classmethod
    def flip_l_r(self, img,depth,width, rdm_state=None):
        flip_left_right = np.flip(img,2)

        if rdm_state is not None:
            random.setstate(rdm_state)
        x = random.randint(0, flip_left_right.shape[2] - width)

        if rdm_state is not None:
            random.setstate(rdm_state)
        z = random.randint(0, flip_left_right.shape[0] - depth)

        if debug is True:
            print('flip_l_r')
            print('x, ', x)
            print('z, ', z)

        crop_img = flip_left_right[z:z+depth, :, x:x+width]

        return crop_img

    @classmethod
    def flip_f_b(self, img,depth,width,rdm_state=None):
        flip_front_back = np.flip(img,0)

        if rdm_state is not None:
            random.setstate(rdm_state)
        x = random.randint(0, flip_front_back.shape[2] - width)

        if rdm_state is not None:
            random.setstate(rdm_state)
        z = random.randint(0, flip_front_back.shape[0] - depth)

        if debug is True:
            print('flip_f_b')
            print('x, ', x)
            print('z, ', z)

        crop_img = flip_front_back[z:z+depth, :, x:x+width]

        return crop_img

    # Rotate 45 90 135 180
    @classmethod
    def Rotate(self, img,degree):
        mask = np.ones_like(img)
        img = ndimage.rotate(img, degree, axes=(0,2),reshape=False)
        mask = ndimage.rotate(mask, degree, axes=(0,2),reshape=False)
        if mask.any() <0.5:
            mask[mask < 0.5] = 0
        else:
            mask[mask > 0.5] = 1
        #print(mask)


        return img, mask

    @classmethod
    def randomCrop(self, img,mask, height, depth, width, rdm_state=None):
        #print(img.shape[0],img.shape[1],img.shape[2])
        assert img.shape[1] == height
        assert img.shape[2] >= width
        assert img.shape[0] >= depth

        if rdm_state is not None:
            random.setstate(rdm_state)
        x = random.randint(0, mask.shape[2] - width)

        if rdm_state is not None:
            random.setstate(rdm_state)
        z = random.randint(0, mask.shape[0] - depth)

        if debug is True:
            print('randomCrop')
            print('x, ', x)
            print('z, ', z)

        crop_mask = mask[z:z+depth, :, x:x+width]

        if crop_mask.all() == 1:
            crop_img = img[z:z+depth, :, x:x+width]
            return crop_img

        else:
            print('pass')
            pass
