import os
import joblib
import glob
import copy 
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import random
import psutil


class batch_train_valid:
    def __init__(self, experiment_type, folder_to_read, train_batch_size, valid_batch_size):
        self.folder_to_read = folder_to_read
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

        self.train_start_i = 0
        self.valid_start_i = 0
        self.train_end_i = self.train_batch_size
        self.valid_end_i = self.valid_batch_size

        self.train, self.valid = self._prepare_data_LFHF(experiment_type)

        self.n_train_batch = len(self.train[0]) // self.train_batch_size
        self.n_valid_batch = len(self.valid[0]) // self.valid_batch_size
        self.batch_train_arr_, self.batch_train_label_arr = self.batch_train()
        self.batch_valid_arr_, self.batch_valid_label_arr = self.batch_valid()

    def _load_LFHF_data(self, train_valid, severity_type):
        """severity_type: mild/moderate/severe/healthy"""
        def _concat(LF, y_LF, HF, y_HF):

            img_LF, img_HF = LF, HF
            print('len(img_LF) {}, len(img_HF) {}'.format(len(img_LF), len(img_HF)))
            assert len(img_LF) == len(img_HF), 'LF and HF images do not have same length'
            #assert (y_LF == y_HF).any(), 'LF and HF labels do not match'

            img_list = list()
            for i in range(len(img_LF)):
                img = np.stack((img_LF[i], img_HF[i]))
                img_list.append(img)
                #print('[_concate] i: {}, free memory: {:.2f}GB'.format(i, psutil.virtual_memory().available / 1e9)
            label_list = y_LF
            return img_list, label_list

        def get_number_of_file(file):
            tmp = glob.glob(file)
            return len(tmp)

        LF_file = self.folder_to_read + '/{}_{}_LF_*.npy'.format(train_valid, severity_type)
        LF_file_n = get_number_of_file(LF_file)
        print('number of file: ', LF_file_n)

        for n in range(LF_file_n):
           LF_file = self.folder_to_read + '/{}_{}_LF_{}.npy'.format(train_valid, severity_type, n)
           print('loading ', LF_file)
           LF_loaded = np.load(LF_file, allow_pickle=True, encoding='bytes')
           if n == 0:
               LF = copy.copy(LF_loaded[0])
               y_LF = copy.copy(LF_loaded[1])
           else:
               LF = np.concatenate((LF, LF_loaded[0]), axis=0)
               y_LF = np.concatenate((y_LF, LF_loaded[1]), axis=0)
           del LF_loaded
       
        HF_file = self.folder_to_read + '/{}_{}_HF_*.npy'.format(train_valid, severity_type)
        HF_file_n = get_number_of_file(HF_file)
        print('number of file: ', HF_file_n)

        for n in range(HF_file_n):
            HF_file = self.folder_to_read + '/{}_{}_HF_{}.npy'.format(train_valid, severity_type, n)
            print('loading ', HF_file)
            HF_loaded = np.load(HF_file, allow_pickle=True, encoding='bytes')
            if n == 0:
               HF = copy.copy(HF_loaded[0])
               y_HF = copy.copy(HF_loaded[1])
            else:
               HF = np.concatenate((HF, HF_loaded[0]), axis=0)
               y_HF = np.concatenate((y_HF, HF_loaded[1]), axis=0)
            del HF_loaded        
        print('LF shape {}, HF shape {}'.format(np.shape(LF), np.shape(HF)))

        train, y_train = _concat(LF, y_LF, HF, y_HF)
        return train, y_train
    
    def save(self, filename, data_to_save):
        with open(filename + '.npy', 'wb') as handle:
             joblib.dump(data_to_save, handle)

    def _prepare_data_LFHF(self, experiment_type):

        def read_all_files(train_valid):
            mild, y_mild = self._load_LFHF_data(train_valid, 'mild')
            self.save('mild', [mild, y_mild])
            del mild, y_mild

            moderate, y_moderate = self._load_LFHF_data(train_valid, 'moderate')
            self.save('moderate', [moderate, y_moderate])
            del moderate, y_moderate
      
            severe, y_severe = self._load_LFHF_data(train_valid, 'severe')
            self.save('severe', [severe, y_severe])
            del severe, y_severe
        
            if experiment_type == 'healthy_vs_disease':
                healthy, y_healthy = self._load_LFHF_data(train_valid, 'healthy')
                self.save('healthy', [healthy, y_healthy])
                del healthy, y_healthy

            return load_all_files()

        def load_all_files():
            mild, y_mild = joblib.load('mild.npy')
            moderate, y_moderate = joblib.load('moderate.npy')            
            X = np.concatenate([mild, moderate], axis=0)
            y = np.concatenate([y_mild, y_moderate], axis=0)
            del mild, y_mild, moderate, y_moderate
            os.remove('mild.npy')
            os.remove('moderate.npy')

            severe, y_severe = joblib.load('severe.npy')            
            X = np.concatenate([X, severe], axis=0)
            print('after concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
            y = np.concatenate([y, y_severe], axis=0)
            del severe, y_severe
            os.remove('severe.npy')

            if experiment_type == 'healthy_vs_disease':
                healthy, y_healthy = joblib.load('healthy.npy')
                print('after concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
                X = np.concatenate([X, healthy], axis=0)
                y = np.concatenate([y, y_healthy], axis=0)
                del healthy, y_healthy
                os.remove('healthy.npy')
            
            train_data = list(zip(X, y))
            random.shuffle(train_data)
            X, y = zip(*train_data)
            return X, y

        train = read_all_files('train')
        valid = read_all_files('valid')
        print('train.shape {}, valid.shape {}'.format(np.shape(train), np.shape(valid)))
        return train, valid

    def batch_train(self):

        assert len(self.train[0]) == len(self.train[1])
        # print('train',len(self.train[0]))
        batch_train_list = list()
        batch_train_label_list = list()
        batch_train_features_list = list()

        for i in range(self.train_start_i, self.train_end_i):
            #            print(i)
            batch_train_list.append(self.train[0][i])
            batch_train_label_list.append(self.train[1][i])

        if self.train_end_i == self.train_batch_size * self.n_train_batch:
            pairs = list(zip(self.train[0], self.train[1]))
            random.shuffle(pairs)
            X, y = zip(*pairs)
            self.train = X, y
            self.train_end_i = 0

        self.train_start_i = self.train_end_i
        self.train_end_i = self.train_start_i + self.train_batch_size

        self.batch_train_arr = np.array(batch_train_list)
        self.batch_train_label_arr = np.array(batch_train_label_list)
        # print(self.batch_train_arr.shape)
        # print(self.batch_train_label_arr.shape)

        # self.batch_train_arr_ = np.transpose(self.batch_train_arr, (0, 2, 3, 4, 1))
        # print(self.batch_train_arr.shape)
        self.batch_train_label_arr = to_categorical(self.batch_train_label_arr, 2)
        # print(self.batch_train_label_arr.shape)

        return self.batch_train_arr, self.batch_train_label_arr

    def batch_valid(self):

        assert len(self.valid[0]) == len(self.valid[1])

        batch_valid_list = list()
        batch_valid_label_list = list()
        batch_valid_features_list = list()

        for i in range(self.valid_start_i, self.valid_end_i):
            batch_valid_list.append(self.valid[0][i])
            batch_valid_label_list.append(self.valid[1][i])

        if self.valid_end_i == self.valid_batch_size * self.n_valid_batch:
            pairs = list(zip(self.valid[0], self.valid[1]))
            random.shuffle(pairs)
            X, y = zip(*pairs)
            self.valid = X, y
            self.valid_end_i = 0

        self.valid_start_i = self.valid_end_i
        self.valid_end_i = self.valid_start_i + self.valid_batch_size

        self.batch_valid_arr = np.array(batch_valid_list)
        self.batch_valid_label_arr = np.array(batch_valid_label_list)

        #self.batch_valid_arr_ = np.transpose(self.batch_valid_arr, (0, 2, 3, 4, 1))
        self.batch_valid_label_arr = to_categorical(self.batch_valid_label_arr, 2)

        return self.batch_valid_arr, self.batch_valid_label_arr
