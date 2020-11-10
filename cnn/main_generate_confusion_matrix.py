import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from statistics import mode


def generate_label(experiment_type):
    if experiment_type == 'healthy_vs_disease':
        return ["healthy", "eczema"]
    elif experiment_type == 'mild_vs_modsev':
        return ["mild", "mod-sev"]


if __name__ == '__main__':
    cv=5
    experiment_type = 'healthy_vs_disease' #'mild_vs_modsev' #'healthy_vs_disease'
    prediction_dir = f"../src/CNN-bottleneck_LFHF/results/CV{cv}/healthy_vs_disease_200/"
    # prediction_dir = f"../src/CNN-bottleneck_LFHF_add_features/results/healthy_v_eczema/CV{cv}/healthy_vs_disease/"
    # prediction_dir = f"../src/CNN-bottleneck_LFHF_add_3features/results/healthy_v_eczema/CV{cv}/healthy_vs_disease/"
    # prediction_dir = f"../src/CNN-bottleneck_LFHF_add_3features/results/CV{cv}/mild_vs_modsev/"
    # prediction_dir = f"../src/CNN-bottleneck_LFHF_add_features/results/CV{cv}/mild_vs_modsev/"

    data_type = 'val'
    if data_type == 'test':
        prediction_file = prediction_dir + 'prediction.csv'
        data_type_col = ""
    elif data_type == 'val':
        prediction_file = prediction_dir + 'val_prediction.csv'
        data_type_col = 'val_'

    prediction_df = pd.read_csv(prediction_file)


    y_pred = []
    y_true = []

    for case in np.unique(prediction_df[f'{data_type_col}case']):
        y_pred_tmp = prediction_df[prediction_df[f'{data_type_col}case'] == case][f'{data_type_col}prediction'].to_list()
        y_truth_tmp = prediction_df[prediction_df[f'{data_type_col}case'] == case][f'{data_type_col}ground_truth'].to_list()
        if y_pred_tmp[1:] == y_pred_tmp[:-1]: # check if all are identifical
            y_pred.append(y_pred_tmp[0])
        else:
            try:
                y_pred.append(mode(y_pred_tmp))
            except:
                print(f'{case} do not have final vote')
                continue

        if y_truth_tmp[1:] == y_truth_tmp[:-1]:  # check if all are identifical
            y_true.append(y_truth_tmp[0])

    labels = generate_label(experiment_type)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f'tn: {tn}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'tp: {tp}')

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    print(accuracy)