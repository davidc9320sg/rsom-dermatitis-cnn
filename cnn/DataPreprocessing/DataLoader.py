import numpy as np
import os
from DataPreprocessing.Crop_Maximum_Height.PrepareData import PrepareData
from DataPreprocessing.Crop_Maximum_Height.Maximum_height import MaxHeight_Three_Preprocess
from DataPreprocessing.Crop_preprocess.n_crop import n_crop
from DataPreprocessing.Concat_LFHF.utils import concat, save
from DataPreprocessing.ExperimentTypes.HealthyVsDisease import HealthyVsDisease
from DataPreprocessing.ExperimentTypes.ThreeSeverities import ThreeSeverities
from DataPreprocessing.ExperimentTypes.MildVsModSev import MildVsModSev
from DataPreprocessing.GetNtimesCrop import GetNtimesCrop
import pandas as pd

class DataLoader:
    def __init__(self, experiment_type):
        self.experiment_type = experiment_type
        self.max_diff        = 374  #[FIXED]
        self.cv_folder       = None
        self.test_folder      = None

        """raw datasets loaded from file without any preprocessing"""
        self.train_datasets  = None
        self.valid_datasets  = None

    def generate_train_valid_data(self, cv):
        self.cv_folder = '../data/healthy_v_eczema/CV{}/'.format(cv)
        # get no_pt information for each cv
        no_pt_data = pd.read_csv(self.cv_folder + 'dataset_description.csv')

        # split classes into four subsets: mild, moderate, and severe, healthy
        prep_data = PrepareData(self.cv_folder + 'coord_train.csv')
        self.train_datasets = prep_data.datasets

        prep_data = PrepareData(self.cv_folder + 'coord_valid.csv')
        self.valid_datasets = prep_data.datasets

        if self.experiment_type == 'healthy_vs_disease':
            self._prepare_data_healthy_vs_disease(no_pt_data)
        if self.experiment_type == 'mild_vs_modsev':
            self._prepare_data_mild_vs_modsev(no_pt_data)
        if self.experiment_type == 'three_severities':
            self._prepare_data_three_severity(no_pt_data)
        return

    def generate_test_data(self):
        self.test_folder = '../data/test/'
        # get no_pt information 
        no_pt_data = pd.read_csv(self.test_folder + 'datasets_description.csv')
        
        # split classes into four subsets: mild, moderate, severe and healthy
        prep_data = PrepareData(self.test_folder + 'coord_test.csv')
        self.test_datasets = prep_data.datasets

        if self.experiment_type == 'healthy_vs_disease':
            self._prepare_data_healthy_vs_disease(no_pt_data, test_data=True)
        if self.experiment_type == 'mild_vs_modsev':
            self._prepare_data_mild_vs_modsev(no_pt_data, test_data=True)
        if self.experiment_type == 'three_severities':
            self._prepare_data_three_severity(no_pt_data, test_data=True)
        return

    def _prepare_flat_data(self, expt_type, sub_outputfolder):

        def _crop_maximum_height_flat_data(data, name):
            """This step crop away black region above and below the skin """
            c_study, c_img, max_diff, upper = data[0], data[2], self.max_diff, data[4],
            preprocess = MaxHeight_Three_Preprocess(c_study, c_img, max_diff, upper, name=name)
            flat_data = preprocess.pre_process_one('R_flat', name)
            data = {
                'flat_data': flat_data
            }
            return data

        datasets = expt_type.datasets
        label_names = expt_type.c_label_names  #mild/moderate/severe
        labels = expt_type.labels  #0/1/2/3, depending on experiment types
        train_valid_test = expt_type.train_valid_test
        ntime_crop = GetNtimesCrop(expt_type)
        n_time_cropping = ntime_crop.n_time_cropping

        for i, (label_name, label) in enumerate(zip(label_names, labels)):
            img_list = _crop_maximum_height_flat_data(datasets['{}'.format(label_name)], name=label_name)
            img_list = img_list['flat_data']
            cp = n_crop(img_list, n_time_cropping['{}'.format(label_name)], c_label=label,
                        outputfolder=sub_outputfolder, output_filename=label_name + '_flat', label_name=label_name,
                        train_valid_test=train_valid_test)
            cp.perform_N_crop()
            del img_list, cp

        return

    def _prepare_LFHF_data(self, expt_type, sub_outputfolder):

        def _crop_maximum_height_LFHF_data(data, name):
            """This step crop away black region above and below the skin """
            c_study, c_img, max_diff, upper = data[0], data[2], self.max_diff, data[4],
            preprocess = MaxHeight_Three_Preprocess(c_study, c_img, max_diff, upper, name=name)
            LF_data = preprocess.pre_process_one('R_flat_LF', name)
            HF_data = preprocess.pre_process_one('R_flat_HF', name)
            data = {
                'LF_data': LF_data,
                'HF_data': HF_data,
            }
            return data

        datasets = expt_type.datasets
        label_names = expt_type.c_label_names  #mild/moderate/severe
        labels = expt_type.labels  #0/1/2/3, depending on experiment types
        train_valid_test = expt_type.train_valid_test
        ntime_crop = GetNtimesCrop(expt_type)
        n_time_cropping = ntime_crop.n_time_cropping_LFHF
 
        for i, (label_name, label) in enumerate(zip(label_names, labels)):
            img_list = _crop_maximum_height_LFHF_data(datasets['{}'.format(label_name)], name=label_name)
            LF_img_list = img_list['LF_data']
            HF_img_list = img_list['HF_data']

            cp = n_crop(LF_img_list, n_time_cropping['{}'.format(label_name)], c_label=label,
                        outputfolder=sub_outputfolder, output_filename=label_name + '_LF', label_name=label_name,
                        train_valid_test=train_valid_test)
            cp.perform_N_crop()
            cp = n_crop(HF_img_list, n_time_cropping['{}'.format(label_name)], c_label=label,
                        outputfolder=sub_outputfolder, output_filename=label_name + '_HF', label_name=label_name,
                        train_valid_test=train_valid_test)
            cp.perform_N_crop()

            del img_list, LF_img_list, HF_img_list, cp
        return
        
    def _prepare_data_healthy_vs_disease(self, no_pt_data, test_data=False):
        if test_data is True:
            sub_testfolder   = self.test_folder + 'healthy_vs_disease' + '/'
            if os.path.exists(sub_testfolder) is False:
                os.mkdir(sub_testfolder)
   
            h_vs_d = HealthyVsDisease(self.test_datasets, no_pt_data, train_valid_test='test')
            # self._prepare_flat_data(h_vs_d, sub_testfolder)
            self._prepare_LFHF_data(h_vs_d, sub_testfolder)
       
        else:
            sub_outputfolder = self.cv_folder + "healthy_vs_disease" + '/'
            if os.path.exists(sub_outputfolder) is False:
               os.mkdir(sub_outputfolder)
       
            h_vs_d = HealthyVsDisease(self.train_datasets, no_pt_data, train_valid_test='train')
            self._prepare_flat_data(h_vs_d, sub_outputfolder)
            # self._prepare_LFHF_data(h_vs_d, sub_outputfolder)
            del h_vs_d
      
            h_vs_d = HealthyVsDisease(self.valid_datasets, no_pt_data, train_valid_test='valid')
            # self._prepare_flat_data(h_vs_d, sub_outputfolder)
            self._prepare_LFHF_data(h_vs_d, sub_outputfolder)
        
        return

    def _prepare_data_three_severities(self, no_pt_data, test_data=False):
        if test_data is True:
            sub_testfolder   = self.test_folder + 'three_severities' + '/'
            if os.path.exists(sub_testfolder) is False:
                os.mkdir(sub_testfolder)
           
            three_severe = ThreeSeverities(self.test_datasets, no_pt_data, train_valid_test='test')
            self._prepare_flat_data(three_severe, sub_testfolder)
            self._prepare_LFHF_data(three_severe, subtestfolder)
       
        else:
            sub_outputfolder = self.cv_folder + "three_severities" + '/'
            if os.path.exists(sub_outputfolder) is False:
                os.mkdir(sub_outputfolder)

            three_severe = ThreeSeverities(self.train_datasets, no_pt_data, train_valid_test='train')
            self._prepare_flat_data(three_severe, sub_outputfolder)
            self._prepare_LFHF_data(three_severe, sub_outputfolder)
            del three_severe
     
            three_severe = ThreeSeverities(self.valid_datasets, no_pt_data, train_valid_test='valid')
            self._prepare_flat_data(three_severe, sub_outputfolder)
            self._prepare_LFHF_data(three_severe, sub_outputfolder)
       
        return

    def _prepare_data_mild_vs_modsev(self, no_pt_data, test_data=False):
        if test_data is True:     
           sub_testfolder   = self.test_folder + 'mild_vs_modsev' + '/'
           if os.path.exists(sub_testfolder) is False:
               os.mkdir(sub_testfolder)
           
           mild_vs_modsev = MildVsModSev(self.test_datasets, no_pt_data, train_valid_test='test')
           # self._prepare_flat_data(mild_vs_modsev, sub_testfolder)
           self._prepare_LFHF_data(mild_vs_modsev, sub_testfolder)

        else:
           sub_outputfolder = self.cv_folder + "mild_vs_modsev" + '/'
           if os.path.exists(sub_outputfolder) is False:
                os.mkdir(sub_outputfolder)

           mild_vs_modsev = MildVsModSev(self.train_datasets, no_pt_data, train_valid_test='train')
           # self._prepare_flat_data(mild_vs_modsev, sub_outputfolder)
           self._prepare_LFHF_data(mild_vs_modsev, sub_outputfolder)
           del mild_vs_modsev
    
           mild_vs_modsev = MildVsModSev(self.valid_datasets, no_pt_data, train_valid_test='valid')
           # self._prepare_flat_data(mild_vs_modsev, sub_outputfolder)
           self._prepare_LFHF_data(mild_vs_modsev, sub_outputfolder)
                     
        return









