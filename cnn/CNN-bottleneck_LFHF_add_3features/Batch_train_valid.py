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
        self.batch_train_arr_, self.batch_train_label_arr, self.batch_train_features_arr = self.batch_train()
        self.batch_valid_arr_, self.batch_valid_label_arr, self.batch_valid_features_arr = self.batch_valid()

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
                #print('[_concate] i: {}, free memory: {:.2f}GB'.format(i, psutil.virtual_memory().available / 1e9))

            label_list = y_LF
            return img_list, label_list

        def get_number_of_file(file):
            tmp = glob.glob(file)
            return len(tmp)

        LF_file = self.folder_to_read + '/{}_{}_LF_*.npy'.format(train_valid, severity_type)
        LF_file_n = get_number_of_file(LF_file)
        print('number of file: ', LF_file_n)

        # if there is no file exists, return None
        if LF_file_n == 0:
           return None, None

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
        #print('X_train shape ', np.shape(train))
        return train, y_train

    def _load_feature_data(self, train_valid, no_sample, severity_type):
        folder_to_read = self.folder_to_read.split('/')
        folder_to_read = '/'.join(folder_to_read[:-1])
        df = pd.read_csv(folder_to_read + '/coord_{}.csv'.format(train_valid), delimiter=',')
        label_name = {
            'mild'    : 0,
            'moderate': 1,
            'severe'  : 2,
            'healthy' : 3
        }

        df_specific_severity = df[df['label'] == label_name[severity_type]]
        feature_df = df_specific_severity[['TEWL_norm', 'TBV_norm', 'LHFR_norm']] 
        n_duplicate = int(no_sample / len(feature_df))
        feature_duplicated = np.tile(feature_df.values, [n_duplicate, 1])
        return feature_duplicated

    def save(self, filename, data_to_save):
        with open(filename + '.npy', 'wb') as handle:
             joblib.dump(data_to_save, handle)
 
    def _prepare_data_LFHF(self, experiment_type):

        def read_all_files(train_valid):
            mild, y_mild = self._load_LFHF_data(train_valid, 'mild')
            self.save('mild', [mild, y_mild])
            img_size = np.shape(mild)[1:]
            no_sample = len(y_mild) if y_mild is not None else 0
            del mild, y_mild

            moderate, y_moderate = self._load_LFHF_data(train_valid, 'moderate')
            self.save('moderate', [moderate, y_moderate])
            no_sample += len(y_moderate) if y_moderate is not None else 0
            del moderate, y_moderate

            severe, y_severe = self._load_LFHF_data(train_valid, 'severe')
            self.save('severe', [severe, y_severe])
            no_sample += len(y_severe) if y_severe is not None else 0
            del severe, y_severe
  
            if experiment_type == 'healthy_vs_disease':
               healthy, y_healthy = self._load_LFHF_data(train_valid, 'healthy')
               self.save('healthy', [healthy, y_healthy])
               no_sample += len(y_healthy) if y_healthy is not None else 0
               del healthy , y_healthy
           
            # allocate memory for X
            desired_shape = (no_sample, ) + img_size
            print('desired shape', desired_shape)
            X = np.empty(desired_shape)            
            return load_all_files(train_valid, X)

        def load_all_files(train_valid, X):

            mild, y_mild = joblib.load('mild.npy', 'r')
            if y_mild is not None:       
                mild_fea = self._load_feature_data(train_valid, len(y_mild), 'mild')
                start, end = 0, len(y_mild)
                X[start:end], y, fea = mild, y_mild, mild_fea
                del mild, y_mild, mild_fea
            print('after mild concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available/1e9)) 
            moderate, y_moderate = joblib.load('moderate.npy', 'r')
            if y_moderate is not None:
                moderate_fea = self._load_feature_data(train_valid, len(y_moderate), 'moderate')
                start, end = copy.copy(end), end + len(y_moderate)
                X[start:end] = moderate
                y = np.concatenate([y, y_moderate], axis=0)
                fea = np.concatenate([fea, moderate_fea], axis=0)
                del moderate, y_moderate, moderate_fea 
            print('after moderate concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available/1e9)) 
 
            severe, y_severe = joblib.load('severe.npy', 'r')
            if y_severe is not None: 
                severe_fea = self._load_feature_data(train_valid, len(y_severe), 'severe')
                start, end = copy.copy(end), end + len(y_severe)
                X[start:end] = severe
                y = np.concatenate([y, y_severe], axis=0)
                fea = np.concatenate([fea, severe_fea], axis=0)
                del severe, y_severe, severe_fea
            print('after severe concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
 
            if experiment_type == 'healthy_vs_disease':
                start = copy.copy(end)
                X[start:], y_healthy = joblib.load('healthy.npy', 'r')
                healthy_fea = self._load_feature_data(train_valid, len(y_healthy), 'healthy')
                y = np.concatenate([y, y_healthy], axis=0)
                fea = np.concatenate([fea, healthy_fea], axis=0)
                del y_healthy, healthy_fea
                os.remove('healthy.npy')
                print('after healthy concat, free memory: {:.2f}GB'.format(psutil.virtual_memory().available / 1e9))
 
            assert len(X) == len(y), 'X and y length do not match'
            assert len(X) == len(fea), 'X and fea length do not match'
            os.remove('mild.npy')
            os.remove('moderate.npy')
            os.remove('severe.npy')
            train_data = list(zip(X, y, fea))
            random.shuffle(train_data)
            X, y, z = zip(*train_data)
            return X, y, z

        train = read_all_files('train')
        valid = read_all_files('valid')
        print('train.shape {}, valid.shape {}'.format(np.shape(train), np.shape(valid)))    
        return train, valid

    def batch_train(self):

        assert len(self.train[0]) == len(self.train[1]) == len(self.train[2])
        # print('train',len(self.train[0]))
        batch_train_list = list()
        batch_train_label_list = list()
        batch_train_features_list = list()

        for i in range(self.train_start_i, self.train_end_i):
            #            print(i)
            batch_train_list.append(self.train[0][i])
            batch_train_label_list.append(self.train[1][i])
            batch_train_features_list.append(self.train[2][i])

        if self.train_end_i == self.train_batch_size * self.n_train_batch:
            pairs = list(zip(self.train[0], self.train[1], self.train[2]))
            random.shuffle(pairs)
            X, y, z = zip(*pairs)
            self.train = X, y, z
            self.train_end_i = 0

        self.train_start_i = self.train_end_i
        self.train_end_i = self.train_start_i + self.train_batch_size

        self.batch_train_arr = np.array(batch_train_list)
        self.batch_train_label_arr = np.array(batch_train_label_list)
        self.batch_train_features_arr = np.array(batch_train_features_list)
        # print(self.batch_train_arr.shape)
        # print(self.batch_train_label_arr.shape)

        # self.batch_train_arr_ = np.transpose(self.batch_train_arr, (0, 2, 3, 4, 1))
        # print(self.batch_train_arr.shape)
        self.batch_train_label_arr = to_categorical(self.batch_train_label_arr, 2)
        # print(self.batch_train_label_arr.shape)

        return self.batch_train_arr, self.batch_train_label_arr, self.batch_train_features_arr

    def batch_valid(self):

        assert len(self.valid[0]) == len(self.valid[1]) == len(self.valid[2])

        batch_valid_list = list()
        batch_valid_label_list = list()
        batch_valid_features_list = list()

        for i in range(self.valid_start_i, self.valid_end_i):
            batch_valid_list.append(self.valid[0][i])
            batch_valid_label_list.append(self.valid[1][i])
            batch_valid_features_list.append(self.valid[2][i])

        if self.valid_end_i == self.valid_batch_size * self.n_valid_batch:
            pairs = list(zip(self.valid[0], self.valid[1], self.valid[2]))
            random.shuffle(pairs)
            X, y, z = zip(*pairs)
            self.valid = X, y, z
            self.valid_end_i = 0

        self.valid_start_i = self.valid_end_i
        self.valid_end_i = self.valid_start_i + self.valid_batch_size

        self.batch_valid_arr = np.array(batch_valid_list)
        self.batch_valid_label_arr = np.array(batch_valid_label_list)
        self.batch_valid_features_arr = np.array(batch_valid_features_list)

        # self.batch_valid_arr_ = np.transpose(self.batch_valid_arr, (0, 2, 3, 4, 1))
        self.batch_valid_label_arr = to_categorical(self.batch_valid_label_arr, 2)

        return self.batch_valid_arr, self.batch_valid_label_arr, self.batch_valid_features_arr

