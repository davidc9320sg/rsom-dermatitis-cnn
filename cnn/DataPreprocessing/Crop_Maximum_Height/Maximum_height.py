import pickle
import numpy as np
from scipy.io import loadmat
# import matplotlib.pyplot as plt
import random
import cv2
import statistics
import psutil

Debug = False


class MaxHeight_Three_Preprocess:

    def __init__(self, c_study, c_img, max_diff, upper,
                 name, no_sample_to_save_start=None, no_sample_to_save_end=None):

        self.c_study = c_study
        self.c_img = c_img
        self.name = name
        self.max_diff = int(max_diff)
        self.upper_ = np.asarray(upper)
        self.upper = self.upper_.astype(int)

        if no_sample_to_save_start is None:
            self.no_sample_to_save_start = 0
        else:
            self.no_sample_to_save_start = no_sample_to_save_start

        if no_sample_to_save_end is None:
            self.no_sample_to_save_end = len(self.c_img)
        else:
            self.no_sample_to_save_end = no_sample_to_save_end

        # self.pre_process_all()

    def pre_process_all(self):
        img_list = self.pre_process_one('R_flat')
        # self.save(self.name + '.npy', img_list)

        img_LF_list = self.pre_process_one('R_flat_LF')
        # self.save(self.name + '_LF.npy', img_LF_list)

        img_HF_list = self.pre_process_one('R_flat_HF')
        # self.save(self.name + '_HF.npy', img_HF_list)

        return img_list, img_LF_list, img_HF_list

    def pre_process_one(self, type, name):
        img_list = list()
        for i in range(self.no_sample_to_save_start, self.no_sample_to_save_end):
            print('Maximum_height {} {}: image{}'.format(type, name, i))
            with open(self.c_img[i], 'rb'):
                print('free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
                print('CPU usage: {:.0f}%'.format(psutil.cpu_percent()))
                if psutil.virtual_memory().available < 2e9: #2GB
                    print('exceed memeory used. Stop process')
                    quit()
                img = self.pre_process(self.c_img[i], type, self.c_study[i], self.upper[i])
                img_list.append(img)

        return img_list

    def save(self, name, data):
        np.save(name, data)
        return

    def pre_process(self, img, type_, study, upper):
        R_flat = loadmat(img).get(type_)
        img = self.crop_3dimg(R_flat, study, upper)
        return img

    def crop_3dimg(self, img, study, upper):
        # img shape H W D
        img = img[upper - 50:upper + self.max_diff, :, :]
        img[:20] = 0  # make hair black

        return img









