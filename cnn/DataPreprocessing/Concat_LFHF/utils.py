import numpy as np
import psutil
import pickle

def concat(LF, HF, start=None, end=None):
       
    img_LF, img_HF = LF[0], HF[0]

    if start is None: 
       start = 0
    if end is None:
       end = len(img_LF) 

    assert len(img_LF) == len(img_HF), 'LF and HF images do not have same length'
    assert (LF[1] == HF[1]).any(),  'LF and HF labels do not match'

    img_list = list()
    for i in range(start, end):    
        img = np.stack((img_LF[i], img_HF[i]))
#        print('img shape', np.shape(img))
#        quit()
        img_list.append(img)
        print('i: {}, psutil.virtual_memory {} '.format(i, psutil.virtual_memory()))
 
    label_list = LF[1]
    return img_list, label_list


def save(filename, data_to_save):
    with open(filename + '.npy', 'wb') as handle:
        pickle.dump(data_to_save, handle)



