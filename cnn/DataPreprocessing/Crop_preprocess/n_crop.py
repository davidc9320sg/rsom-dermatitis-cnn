import pickle
import numpy as np
#import matplotlib.pyplot as plt
from .crop_rotate import crop_rotate_utils
from .action_done import action_done
import cv2 as cv
import random
from scipy import ndimage
import csv
import psutil
import os
from .SaveData import SaveData


class n_crop(crop_rotate_utils):
    def __init__(self, c_img, c_num, c_label, outputfolder=None, output_filename=None, label_name=None,
                 train_valid_test=None, img_idx_to_start=None, img_idx_to_end=None):
        self.action_done = action_done()
        self.c_img = c_img
        self.c_label = str(c_label)
        self.c_num = c_num
        self.outputfolder = outputfolder
        self.output_filename = output_filename  #mild/mild_LF/mild_HF
        self.label_name = label_name  #mild/moderate/severe/healthy
        self.train_valid_test = train_valid_test

        rdm_state_file = self.outputfolder + '/' + self.label_name + '_random_state' + '.pkl'
        if os.path.isfile(rdm_state_file) is True:
            print('loading ' + rdm_state_file)
            tmp = np.load(rdm_state_file, allow_pickle=True)
            self.action_list = tmp['action_done']
            self.rdm_state_list = tmp['random_state']
            self.save_random_state = False
        else:
            self.action_list = None
            self.rdm_state_list = None
            self.save_random_state = True

        if img_idx_to_start is None:
           self.start = 0
        else:
           self.start = img_idx_to_start

        if img_idx_to_end is None:
           self.end = len(self.c_img)
        else:
           self.end = img_idx_to_end

    def eval_save_random_state(self):
        print('save random state')
        if self.save_random_state is True:
            with open(self.outputfolder + '/' + self.label_name + '_random_state' + '.pkl', 'wb') as handle:
                pickle.dump(self.action_done.action_rdmstate, handle)

    @staticmethod
    def normalize(img, tmin, tmax):
        xmax, xmin = img.max(), img.min()
        img_ = (img - xmin) / (xmax - xmin) * (tmax -tmin) + tmin
        return img_        

    def perform_N_crop(self):
        LFHF = self.output_filename.lstrip(self.label_name + '_')
        save_data = SaveData(self.outputfolder, self.train_valid_test, self.output_filename)

        k=0
        count = None
        for i in range(self.start, self.end):
            print('-----------------------------------------------------------------')
            print('[{}] Ncrop {} image {}'.format(self.train_valid_test, self.output_filename, i))
            print('free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
            print('CPU usage: {:.0f}%'.format(psutil.cpu_percent()))

            if psutil.virtual_memory().available < 2e9:  # 2GB
                print('exceed memeory used. Stop process')
                quit()

            # img shape H W D
            img = self.c_img[i]

            img = self.normalize(img, tmin=0, tmax=1)
            
            img = np.moveaxis(img, -1, 0)

            actions = ['flip_f_b','flip_l_r','rotate0','rotate45','rotate90','rotate135','rotate180']
            random.shuffle(actions)
            count = 0
            for j in range(500):
                if count == self.c_num:
                    break

               #  print('k ' , k)
                if self.action_list is None or self.rdm_state_list is None:
                    action = random.choice(actions)
                    rdm_state = random.getstate()
                    self.action_done.append(action, random.getstate())
                    random.seed(random.randint(0, 1e8))
                else:
                    action = self.action_list[k]
                    rdm_state = self.rdm_state_list[k]

                print('action', action)

                if action == 'flip_f_b':
                    crop_img = super().flip_f_b(img,64,64, rdm_state)

                    save_data.append(crop_img, self.c_label, LFHF)
                    count += 1

                elif action == 'flip_l_r':
                    crop_img = super().flip_l_r(img,64,64,rdm_state)
                    save_data.append(crop_img, self.c_label, LFHF)
                    count += 1

                elif action == 'rotate0':
                    rotate_img, rotate_mask = crop_rotate_utils.Rotate(img,0)
                    crop_img = crop_rotate_utils.randomCrop(rotate_img,rotate_mask,rotate_img.shape[1],64,64,rdm_state)
                    try:
                        save_data.append(crop_img, self.c_label, LFHF)
                        count += 1
                    except AttributeError:
                        del self.action_done.action_rdmstate['action_done'][-1]
                        del self.action_done.action_rdmstate['random_state'][-1]
                        print('shape not found')

                elif action == 'rotate45':
                    rotate_img, rotate_mask = crop_rotate_utils.Rotate(img,45)
                    crop_img = crop_rotate_utils.randomCrop(rotate_img,rotate_mask,rotate_img.shape[1],64,64,rdm_state)
                    try:
                        crop_img.shape
                        save_data.append(crop_img, self.c_label, LFHF)
                        count += 1
                    except AttributeError:
                        del self.action_done.action_rdmstate['action_done'][-1]
                        del self.action_done.action_rdmstate['random_state'][-1]     
                        print('shape not found')

                elif action == 'rotate90':
                    rotate_img, rotate_mask = crop_rotate_utils.Rotate(img,90)
                    crop_img = crop_rotate_utils.randomCrop(rotate_img,rotate_mask,rotate_img.shape[1],64,64,rdm_state)
                    try:
                        crop_img.shape
                        save_data.append(crop_img, self.c_label, LFHF)
                        count += 1
                    except AttributeError:
                        del self.action_done.action_rdmstate['action_done'][-1]
                        del self.action_done.action_rdmstate['random_state'][-1]     
                        print('shape not found')

                elif action == 'rotate135':
                    rotate_img, rotate_mask = crop_rotate_utils.Rotate(img,135)
                    crop_img = crop_rotate_utils.randomCrop(rotate_img,rotate_mask,rotate_img.shape[1],64,64,rdm_state)
                    try:
                        crop_img.shape
                        save_data.append(crop_img, self.c_label, LFHF)
                        count += 1
                    except AttributeError:
                        del self.action_done.action_rdmstate['action_done'][-1]
                        del self.action_done.action_rdmstate['random_state'][-1]     
                        print('shape not found')
 
                else:
                    if action == 'rotate180':
                        rotate_img, rotate_mask = crop_rotate_utils.Rotate(img,180)
                        crop_img = crop_rotate_utils.randomCrop(rotate_img,rotate_mask,rotate_img.shape[1],64,64,rdm_state)
                        try:
                            crop_img.shape
                            save_data.append(crop_img, self.c_label, LFHF)
                            count += 1
                        except AttributeError:
                            del self.action_done.action_rdmstate['action_done'][-1]
                            del self.action_done.action_rdmstate['random_state'][-1]     
                            print('shape not found')
 
                k += 1
        print('count: ', count)

        if count is None or count == 0:
            return
        else:
            save_data.save_model(save_model=True)
            self.eval_save_random_state() 
        
        return
