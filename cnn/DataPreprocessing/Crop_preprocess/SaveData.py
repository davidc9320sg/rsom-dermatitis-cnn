import pickle
import psutil
import os
import glob


class SaveData:
    def __init__(self, outputfolder, train_valid_test, filename):
        self.img_list = []
        self.img_label = []
        self.outputfolder = outputfolder
        self.train_valid_test = train_valid_test  #train/valid/test
        self.filename = filename  #mild/mild_LF/mild_HF
        self.img_list_length_to_save = 0
        self.initial_memory = psutil.virtual_memory().available

    def append(self, img_list, img_label, LFHF):
        self.img_list.append(img_list)
        self.img_label.append(img_label)
        self.save_model(LFHF)

    def save_model(self, save_model=False, LFHF='LF'):
        memory_used = self.initial_memory - psutil.virtual_memory().available
        memory_available = psutil.virtual_memory().available
        # print('memory used {:.2f}Gb / memory_available {:.2f}Gb'.format(memory_used/1e9, memory_available/1e9))
        if save_model is True:
            filename = self.get_filename(self.outputfolder, self.train_valid_test, self.filename)
            self.save(filename, [self.img_list, self.img_label])
            self.img_list, self.img_label = [], []
        else:
            if LFHF == 'LF':
                if memory_used * 2 > memory_available:
                    filename = self.get_filename(self.outputfolder, self.train_valid_test, self.filename)
                    self.save(filename, [self.img_list, self.img_label])
                    self.img_list, self.img_label = [], []
                    self.img_list_length_to_save = len(self.img_list)
            if LFHF == 'HF':
                if len(self.img_list) == self.img_list_length_to_save:
                    filename = self.get_filename(self.outputfolder, self.train_valid_test, self.filename)
                    self.save(filename, [self.img_list, self.img_label])
                    self.img_list, self.img_label = [], []
                    self.img_list_length_to_save = len(self.img_list)
            return

    def get_filename(self, outputfolder, train_valid_test, filename):
        pathname = '{}{}_{}*.npy'.format(outputfolder, train_valid_test, filename)
        tmp = glob.glob(pathname)
        if len(tmp) == 0:
            k = 0
        else:
            k = len(tmp)
        output_full_filname = '_'.join([self.train_valid_test, self.filename, str(k)])
        print('output_full_filename', output_full_filname)
        return self.outputfolder + output_full_filname

    def save(self, filename, data_to_save):
        print('save file into ', filename)
        with open(filename + '.npy', 'wb') as handle:
            pickle.dump(data_to_save, handle)
