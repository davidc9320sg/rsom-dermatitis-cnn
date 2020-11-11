"""This code is to generate additional sample for prediction if that patient does not have final prediction"""

import pickle
import random 
import numpy as np
import pandas as pd
from scipy.io import loadmat
import argparse
from DataPreprocessing.Crop_preprocess.crop_rotate import crop_rotate_utils

def normalize(img, tmin, tmax):
    xmax, xmin = img.max(), img.min()
    img_ = (img - xmin) / (xmax - xmin) * (tmax- tmin) + tmin
    return img_


MAX_DIFF = 374 #[FIXED]

if __name__ == '__main__':
    # parser argument 
    parser = argparse.ArgumentParser()
    parser.add_argument('study_case_number', type=str, help='key in the case number you wish to augment. Eg: study0040')
    parser.add_argument('output_folder', type=str, help='Eg: ~/rsom-bii-sbic/data/CV3/mild_vs_modsev/')
    parser.add_argument('data_type', type=str, help='Eg. test / valid / train')
    args = parser.parse_args()
    
    study_case_number = args.study_case_number
    output_folder = args.output_folder
    train_valid_test = args.data_type
    print('case number: {}'.format(study_case_number))
    print('output_folder: {}'.format(output_folder))

    # read csv file to get images informations
    train = pd.read_csv('../data/features_norm_train.csv')
    valid = pd.read_csv('../data/features_norm_valid.csv')
    csv_data = pd.concat([train, valid], axis=0)
    print(csv_data.head())

    patient_row = csv_data[csv_data['case'] == study_case_number]
    study = str(patient_row['case'].values[0])
    label = str(patient_row['label'].values[0])
    file_dir = str(patient_row['direc'].values[0])
    lower = int(patient_row['lower_coords'].values[0])
    upper = int(patient_row['upper_coords'].values[0])
    diff = int(patient_row['difference'].values[0])
    
    # load images
    LF_img = loadmat(file_dir).get('R_flat_LF')
    HF_img = loadmat(file_dir).get('R_flat_HF')
    print('LF_img.shape ', LF_img.shape)
    print('HF_img.shape ', HF_img.shape)
  
    # crop maximum 
    LF_img = LF_img[upper - 50:upper + MAX_DIFF, :, :]
    LF_img[:20] = 0
    HF_img = HF_img[upper - 50:upper + MAX_DIFF, :, :]
    HF_img[:20] = 0
    print('after crop_max')
    print('LF_img.shape ', LF_img.shape)
    print('HF_img.shape ', HF_img.shape)

    # normalize
    LF_img = normalize(LF_img, tmin=0, tmax=1)
    HF_img = normalize(HF_img, tmin=0, tmax=1)
    
    # moveaxis
    LF_img = np.moveaxis(LF_img, -1, 0)
    HF_img = np.moveaxis(HF_img, -1, 0)

    # augmentation
    actions = ['flip_f_b', 'flip_l_r', 'rotate0', 'rotate_45', 'rotate90', 'rotate135', 'rotate180']
    random.shuffle(actions)

    count = 0
    img_list, img_label = [], []
    for i in range(20):
        if count == 1:
            break

        action = random.choice(actions)
        print(action)

        if action == 'flip_f_b':
            LF_crop_img = crop_rotate_utils.flip_f_b(LF_img, 64, 64)
            HF_crop_img = crop_rotate_utils.flip_f_b(HF_img, 64, 64)
            LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
            count += 1
       
        elif action == 'flip_l_r':
            LF_crop_img = crop_rotate_utils.flip_l_r(LF_img, 64, 64)
            HF_crop_img = crop_rotate_utils.flip_l_r(HF_img, 64, 64)
            LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
            count += 1
        
        elif action == 'rotate0':
            LF_rotate_img, LF_rotate_mask = crop_rotate_utils.Rotate(LF_img, 0)
            HF_rotate_img, HF_rotate_mask = crop_rotate_utils.Rotate(HF_img, 0)
            LF_crop_img = crop_rotate_utils.randomCrop(LF_rotate_img, LF_rotate_mask, LF_rotate_img.shape[1], 64, 64) 
            HF_crop_img = crop_rotate_utils.randomCrop(HF_rotate_img, HF_rotate_mask, HF_rotate_img.shape[1], 64, 64)
            
            if LF_crop_img is None or HF_crop_img is None:
               print('shape not found')
               continue
            else:
               LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
               count += 1
         
        elif action == 'rotate45':
            LF_rotate_img, LF_rotate_mask = crop_rotate_utils.Rotate(LF_img, 45)
            HF_rotate_img, HF_rotate_mask = crop_rotate_utils.Rotate(HF_img, 45)
            LF_crop_img = crop_rotate_utils.randomCrop(LF_rotate_img, LF_rotate_mask, LF_rotate_img.shape[1], 64, 64) 
            HF_crop_img = crop_rotate_utils.randomCrop(HF_rotate_img, HF_rotate_mask, HF_rotate_img.shape[1], 64, 64)
            
            if LF_crop_img is None or HF_crop_img is None:
               print('shape not found')
               continue
            else:
               LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
               count += 1
                  
        elif action == 'rotate90':
            LF_rotate_img, LF_rotate_mask = crop_rotate_utils.Rotate(LF_img, 90)
            HF_rotate_img, HF_rotate_mask = crop_rotate_utils.Rotate(HF_img, 90)
            LF_crop_img = crop_rotate_utils.randomCrop(LF_rotate_img, LF_rotate_mask, LF_rotate_img.shape[1], 64, 64) 
            HF_crop_img = crop_rotate_utils.randomCrop(HF_rotate_img, HF_rotate_mask, HF_rotate_img.shape[1], 64, 64)
            
            if LF_crop_img is None or HF_crop_img is None:
                print('shape not found')
                continue
            else:
               LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
               count += 1
         
        elif action == 'rotate135':
            LF_rotate_img, LF_rotate_mask = crop_rotate_utils.Rotate(LF_img, 135)
            HF_rotate_img, HF_rotate_mask = crop_rotate_utils.Rotate(HF_img, 135)
            LF_crop_img = crop_rotate_utils.randomCrop(LF_rotate_img, LF_rotate_mask, LF_rotate_img.shape[1], 64, 64) 
            HF_crop_img = crop_rotate_utils.randomCrop(HF_rotate_img, HF_rotate_mask, HF_rotate_img.shape[1], 64, 64)
           
            if LF_crop_img is None or HF_crop_img is None:
               print('shape not found')
               continue
            else:
               LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
               count += 1

        elif action == 'rotate180':
            LF_rotate_img, LF_rotate_mask = crop_rotate_utils.Rotate(LF_img, 180)
            HF_rotate_img, HF_rotate_mask = crop_rotate_utils.Rotate(HF_img, 180)
            LF_crop_img = crop_rotate_utils.randomCrop(LF_rotate_img, LF_rotate_mask, LF_rotate_img.shape[1], 64, 64) 
            HF_crop_img = crop_rotate_utils.randomCrop(HF_rotate_img, HF_rotate_mask, HF_rotate_img.shape[1], 64, 64)
            
            if LF_crop_img is None or HF_crop_img is None:
               print('shape not found')
               continue
            else:
               LFHF_crop_img = np.stack((LF_crop_img, HF_crop_img))
               count += 1

    print(LFHF_crop_img.shape)
    
    
    # save to npy file
    data_to_save = [LFHF_crop_img, label]
    filename = output_folder + study_case_number + '_LFHF_' + train_valid_test
    print('save file into ', filename)
    with open(filename + '.npy', 'wb') as handle:
        pickle.dump(data_to_save, handle)
