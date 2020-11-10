import os
import random as rn
from Batch_train_valid import batch_train_valid
from keras.utils import to_categorical
import pandas as pd
import copy
from Build_Model import CNN
import glob
import tensorflow as tf
import numpy as np


class PrepareData:
    def __init__(self, train_valid_test, experiment_type, folder_to_read):
        self.train_valid_test = train_valid_test #train/valid/test data
        self.folder_to_read = folder_to_read
        additional_data = self._load_additional_npy(experiment_type)
        data = self._prepare_data_LFHF(experiment_type)
        
        if additional_data[0] is not None:
           X = np.vstack((data[0], additional_data[0]))
           y = np.vstack((data[1], additional_data[1]))
           fea = np.vstack((data[2], additional_data[2]))
           casename = np.hstack((data[3], additional_data[3]))
           self.data = X, y, fea, casename

        else:
           self.data = data

    def assign_label(self, label,  experiment_type):
        """0-mild, 1-moderate, 2-severe, 3-healthy"""
        if experiment_type == 'healthy_vs_disease':
           y = 0 if (label == '0') or (label == '1') or (label == '2') else 1

        if experiment_type == 'mild_vs_modsev':
           y = 0 if (label == '0') else 1 
        
        return y

    def _load_additional_npy(self, experiment_type):
        study_file = self.folder_to_read + 'study*{}.npy'.format(self.train_valid_test)
        study_file_list = glob.glob(study_file)
        
        if len(study_file_list)  == 0:
           return None, None, None, None
      
        # load features csv
        folder_to_read = os.path.dirname(os.path.dirname(self.folder_to_read))
        fea_df = pd.read_csv(folder_to_read + '/coord_{}.csv'.format(self.train_valid_test))

        img_list, label_list, fea_list, casename_list = [], [], [], []
        for LFHF_file_tmp in study_file_list:
            print('loading ', LFHF_file_tmp)
            LFHF_loaded = np.load(LFHF_file_tmp, allow_pickle=True, encoding='bytes')
            img, label = LFHF_loaded[0], LFHF_loaded[1]
            label = self.assign_label(label, experiment_type)
            casename = LFHF_file_tmp.split('/')[-1].split('.')[0].split('_')[0]
            fea = fea_df[fea_df['case'] == casename].iloc[:, -3:].values.squeeze()
            img_list.append(img)
            label_list.append(label)
            fea_list.append(fea)
            casename_list.append(casename)

        label_list = to_categorical(np.array(label_list), 2)

        return img_list, label_list, fea_list, casename_list
        

    def _load_LFHF_data(self, severity_type):
        def _get_number_of_file(file):
            tmp = glob.glob(file)
            return len(tmp)
        
        def _concat(LF, y_LF, HF, y_HF):
            img_LF, img_HF = LF, HF
            print('len(img_LF) {}, len(img_HF) {}'.format(len(img_LF), len(img_HF)))
            assert len(img_LF) == len(img_HF), 'LF and HF images do not have same length'
           
            img_list = list()
            for i in range(len(img_LF)):
                img = np.stack((img_LF[i], img_HF[i]))
                img_list.append(img)
            label_list = y_LF
            return img_list, label_list 
         
        LF_file = self.folder_to_read + '{}_{}_LF*.npy'.format(self.train_valid_test, severity_type)
        LF_file_n = _get_number_of_file(LF_file)
        print('number of file: ', LF_file_n)
       
        # if there is no file exists, return None
        if LF_file_n == 0:
            return None, None
    
        for n in range(LF_file_n):
            LF_file = self.folder_to_read + '/{}_{}_LF_{}.npy'.format(self.train_valid_test, severity_type, n)
            print('loading ', LF_file)
            LF_loaded = np.load(LF_file, allow_pickle=True, encoding='bytes')
            if n == 0:
                LF = copy.copy(LF_loaded[0])
                y_LF = copy.copy(LF_loaded[1])
            else:
                LF = np.concatenate((LF, LF_loaded[0]), axis=0)
                y_LF = np.concatenate((y_LF, LF_loaded[1]), axis=0)
            del LF_loaded
        
        HF_file = self.folder_to_read + '{}_{}_HF*.npy'.format(self.train_valid_test, severity_type)
        HF_file_n = _get_number_of_file(HF_file)
        print('number of file: ', HF_file_n)
    
        for n in range(HF_file_n):
            HF_file = self.folder_to_read + '/{}_{}_HF_{}.npy'.format(self.train_valid_test, severity_type, n)
            print('loading ', LF_file)
            HF_loaded = np.load(HF_file, allow_pickle=True, encoding='bytes')
            if n == 0:
                HF = copy.copy(HF_loaded[0])
                y_HF = copy.copy(HF_loaded[1])
            else:
                HF = np.concatenate((HF, HF_loaded[0]), axis=0)
                y_HF = np.concatenate((y_HF, HF_loaded[1]), axis=0)
            del HF_loaded
    
        print('LF shape {}, HF shape {}'.format(np.shape(LF), np.shape(HF)))
        
        test, y_test = _concat(LF, y_LF, HF, y_HF)
        return test, y_test

    def _load_feature_data(self, no_sample, severity_type):
        folder_to_read = self.folder_to_read.split('/')
        folder_to_read = '/'.join(folder_to_read[:-2])
        df = pd.read_csv(folder_to_read + '/coord_{}.csv'.format(self.train_valid_test, delimiter=','))
        label_name = {
            'mild'    : 0,
            'moderate': 1,
            'severe'  : 2,
            'healthy' : 3
        }

        df_specific_severity = df[df['label']==label_name[severity_type]]
        feature_df = df_specific_severity.iloc[:, -3:]
        case_name  = df_specific_severity['case']
        n_time_duplicate = int(no_sample / len(feature_df))
        feature_duplicated = np.tile(feature_df.values, [n_time_duplicate, 1])
        case_name_duplicated = np.tile(case_name.values, [n_time_duplicate, 1]).ravel()
        return feature_duplicated, case_name_duplicated

    def _prepare_data_LFHF(self, experiment_type):
        mild, y_mild = self._load_LFHF_data('mild')
        if y_mild is not None:
            mild_fea, mild_casename = self._load_feature_data(len(y_mild), 'mild')
            img_size = np.shape(mild)[1:]
            no_sample = len(y_mild) if y_mild is not None else 0
 
        moderate, y_moderate = self._load_LFHF_data('moderate')
        if y_moderate is not None:    
            moderate_fea, moderate_casename = self._load_feature_data(len(y_moderate), 'moderate')
            no_sample += len(y_moderate) if y_moderate is not None else 0
           
        severe, y_severe = self._load_LFHF_data('severe')
        if y_severe is not None:
            severe_fea, severe_casename = self._load_feature_data(len(y_severe), 'severe')
            no_sample += len(y_severe) if y_severe is not None else 0
 
        if experiment_type == 'healthy_vs_disease':
            healthy, y_healthy = self._load_LFHF_data('healthy')
            if y_healthy is not None:
                healthy_fea, healthy_casename = self._load_feature_data(len(y_healthy), 'healthy')
                no_sample += len(y_healthy) if y_healthy is not None else 0 
        
        desired_shape = (no_sample, ) + img_size
        print('desired shape', desired_shape)
        
        # allocate memory to X and fill it with mild, moderate, severe
        # do this way to save memory. concatenation used twice memory
        X = np.empty(desired_shape)
        
        if y_mild is not None:
            start, end = 0, len(y_mild)
            X[start:end], y, fea, casename = mild, y_mild, mild_fea, mild_casename
            del mild, y_mild, mild_fea, mild_casename

        if y_moderate is not None:
            start, end = copy.copy(end), end + len(y_moderate)
            X[start:end] = moderate
            y = np.concatenate([y, y_moderate], axis=0)
            fea = np.concatenate([fea, moderate_fea], axis=0)
            casename = np.concatenate([casename, moderate_casename], axis=0)
            del moderate, y_moderate, moderate_fea, moderate_casename
        
        if y_severe is not None:
            start, end = copy.copy(end), end + len(y_severe)
            X[start:end] = severe
            y = np.concatenate([y, y_severe], axis=0)
            fea = np.concatenate([fea, severe_fea], axis=0)
            casename = np.concatenate([casename, severe_casename], axis=0)
            del severe, y_severe, severe_fea, severe_casename
  
        if experiment_type == 'healthy_vs_disease':
            if y_healthy is not None:
                start = copy.copy(end)
                X[start:] = healthy
                y = np.concatenate([y, y_healthy], axis=0)
                fea = np.concatenate([fea, healthy_fea], axis=0)
                casename = np.concatenate([casename, healthy_casename], axis=0)
                del healthy, y_healthy, healthy_fea, healthy_casename 
             
        assert len(X) == len(y), 'X and y length do not match'
        assert len(X) == len(fea), 'X and feature length do not match'
       
        X = np.array(X)
        y = to_categorical(np.array(y), 2)
        fea = np.array(fea)
        return X, y, fea, casename



if __name__ == '__main__':
    # model_epoch_to_load
    cv = 3 
    epoch_to_load = 30
    experiment_type =  'healthy_vs_disease' #'mild_vs_modsev'     
    test_folder_to_read = '../../data/test/{}/'.format(experiment_type) 
    valid_folder_to_read = '../../data/CV{}/{}_200/'.format(cv, experiment_type)
    loss_cv_folder = 'results/CV{}/{}_200/'.format(cv, experiment_type) 
    checkpoint_dir = 'results/CV{}/{}_200/model/my_model.ckpt-{}'.format(cv, experiment_type, epoch_to_load)
    output_folder = loss_cv_folder

    # load testing data
    TEST_BATCH_SIZE = 4
    load_test_data = PrepareData('test', experiment_type, test_folder_to_read)
    X_test, y_test, fea_test, test_casename = load_test_data.data

    # load valid data to check if loaded correctly or not
    VALID_BATCH_SIZE = 4
    load_valid_data = PrepareData('valid', experiment_type, valid_folder_to_read)
    X_valid, y_valid, fea_valid, valid_casename = load_valid_data.data
    loaded_loss_acc_csv = pd.read_csv(loss_cv_folder + 'loss_curve.txt', header=None, delimiter=' ')
    loaded_loss_acc_csv.columns = ['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss']
    loaded_epoch_val_acc = loaded_loss_acc_csv[loaded_loss_acc_csv['epoch'] == epoch_to_load]['val_acc'].values

    # specific dimension for cnn
    n_features = fea_test.shape[1]
    channels, width, height, depth = X_test.shape[1:]
    if experiment_type == 'healthy_vs_disease' or experiment_type == 'mild_vs_modsev':
        prediction_output = 2
    elif experiment_type == 'three_severities':
        prediction_output = 3


    # create placeholder
    X = tf.placeholder(tf.float32, shape=[None, channels, width, height, depth], name='X')
    y = tf.placeholder(tf.float32, shape=[None, prediction_output], name='Y')
    #features = tf.placeholder(tf.float32, shape=[None, n_features], name='ft')

    # build model
    CNN_name = "rsom"
    my_cnn = CNN(CNN_name, prediction_output)
    model_output = my_cnn._build_model(X)
    model_saver = tf.train.Saver()

    # specify accuracy
    comp_pred = tf.equal(tf.argmax(y, 1), tf.argmax(model_output, 1))
    accuracy = tf.reduce_mean(tf.cast(comp_pred, tf.float32))
   
    with tf.Session() as sess:
        loaded_model = model_saver.restore(sess, checkpoint_dir)

        # ============= MAKE PREDICTION FOR VALID DATA  ====================#
        # CHECK IF CORRECT
        epoch_v_acc = 0
        val_prediction_one_hot_list = []
        val_prediction_int_list = []
        for i in range(len(X_valid) // VALID_BATCH_SIZE):
            start, end = i * VALID_BATCH_SIZE, (i+1) * VALID_BATCH_SIZE
            if len(X_valid) - end < VALID_BATCH_SIZE:
                end = len(X_valid)
            X_valid_tmp = X_valid[start:end]
            y_valid_tmp = y_valid[start:end]
            fea_valid_tmp = fea_valid[start:end]                        
            val_prediction_one_hot = sess.run(model_output, feed_dict={X:X_valid_tmp})
            val_prediction_int = np.argmax(val_prediction_one_hot, axis=1)
            val_prediction_one_hot_list.append(val_prediction_one_hot)
            val_prediction_int_list.append(val_prediction_int)
        
            val_acc = sess.run([accuracy], feed_dict={X:X_valid_tmp, y:y_valid_tmp})
            epoch_v_acc += np.asarray(val_acc)

        epoch_val_acc = epoch_v_acc / (len(X_valid)/VALID_BATCH_SIZE)
        print('val_acc computed: {}'.format(epoch_val_acc))
        print('Model loaded correctly')
        
        # ============= MAKE PREDICTION FOR TEST DATA ======================= #
        test_prediction_one_hot_list = []
        test_prediction_int_list     = []
        for n in range(len(X_test) // TEST_BATCH_SIZE):
            start, end = n * TEST_BATCH_SIZE, (n+1) * TEST_BATCH_SIZE
            if len(X_test) - end < TEST_BATCH_SIZE:
                end = len(X_test)
            X_test_tmp = X_test[start:end]
            fea_test_tmp = fea_test[start:end] 
            test_prediction_one_hot = sess.run(model_output, feed_dict={X:X_test_tmp})
            test_prediction_int = np.argmax(test_prediction_one_hot, axis=1)
            test_prediction_one_hot_list.append(test_prediction_one_hot)
            test_prediction_int_list.append(test_prediction_int)

    val_prediction_one_hot_arr = np.vstack(val_prediction_one_hot_list)
    val_prediction_int_arr = np.hstack(val_prediction_int_list)
    val_y_ground_truth = np.argmax(y_valid, axis=1)

    test_prediction_one_hot_arr = np.vstack(test_prediction_one_hot_list)
    test_prediction_int_arr     = np.hstack(test_prediction_int_list)
    test_y_ground_truth         = np.argmax(y_test, axis=1)
    
    val_df = pd.DataFrame(columns=['val_case', 'val_softmax_0', 'val_softmax_1', 'val_prediction', 'val_ground_truth'])
    test_df = pd.DataFrame(columns=['case', 'softmax_0', 'softmax_1', 'prediction', 'ground_truth'])

    val_df['val_case'] = valid_casename
    val_df[['val_softmax_0', 'val_softmax_1']] = val_prediction_one_hot_arr
    val_df['val_prediction'] = val_prediction_int_arr
    val_df['val_ground_truth'] = val_y_ground_truth

    test_df['case'] = test_casename
    test_df[['softmax_0', 'softmax_1']] = test_prediction_one_hot_arr
    test_df['prediction'] = test_prediction_int_arr
    test_df['ground_truth'] = test_y_ground_truth

    print(val_df)

    quit()
    val_df.to_csv(output_folder + 'val_prediction.csv', index=False)
    # test_df.to_csv(output_folder + 'prediction.csv', index=False)

        

