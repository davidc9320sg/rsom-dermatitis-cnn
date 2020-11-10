from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pandas as pd
import numpy as np
from PrepareData import PrepareData
from SJ_code.Preprocess.XT_Preprocess import XT_Preprocess
import matplotlib.pyplot as plt


def process_all_file():
    load = PrepareData()  # split classes into three subsets: mild, moderate, and severe
    # study, labels, dirs
    mild = load.mild
    moderate = load.moderate
    severe = load.severe

    print('mild')
    p_mild = XT_Preprocess(mild[2], mild[1])
    p_mild.pre_process_all()
    print('moderate')
    p_moderate = XT_Preprocess(moderate[2], moderate[1])
    p_moderate.pre_process_all()
    print('severe')
    p_severe = XT_Preprocess(severe[2], severe[1])
    p_severe.pre_process_all()

    columns = ['case', 'label', 'direc', 'skin_surface_coord', 'good/bad']
    df = pd.DataFrame(columns=columns)
    df["case"] = np.concatenate((mild[0], moderate[0], severe[0]))
    df['label'] = np.concatenate((mild[1], moderate[1], severe[1]))
    df['direc'] = np.concatenate((mild[2], moderate[2], severe[2]))

    df['skin_surface_coord'] = np.concatenate((p_mild.bounding_box[:, 0], p_moderate.bounding_box[:, 0],
                                               p_severe.bounding_box[:, 0]))
    df['good/bad'] = np.concatenate((p_mild.feedback, p_moderate.feedback, p_severe.feedback))

    # df.to_csv('bounding_box.csv', index=False, index_label=False)
    return


def process_one(casename):
    pp = XT_Preprocess(casename, 1)
    pp.pre_process(casename)


if __name__ == '__main__':
    # casename = "E:\BII\Shiernee\RSOM\data\study0081\R_053838_058_05092018_1730_2_2_mc1\mat_flat.mat"
    # process_one(casename)
    process_all_file()


