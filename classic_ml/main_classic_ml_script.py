import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score
import pandas as pd
from os import makedirs


def group_severity(x):
    if x == 1 or x == 2:
        return 1
    elif x == 0:
        return 0
    else:
        raise Exception("labels must be 0, 1 or 2.")

def group_healthy_v_ecz(x):
    if x == 3:
        return 0
    else:
        return 1

def evaluate_results(classifier, features, y_true):
    y_pred = classifier.predict(features)
    cmat = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    rec = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    return {'cmat': cmat, 'acc': acc, 'rec': rec, 'prec': prec}


if __name__ == '__main__':
    # NOTE on labels: 0:mild, 1:moderate, 2:severe, 3:healthy
    plt.close('all')
    clf_list = ['SVM', 'RF']
    feats_strat = [
        ['TBV_norm', 'LHFR_norm', 'TEWL_norm'],
        ['ET_norm', 'TBV_norm', 'LHFR_norm', 'TEWL_norm'],
        ['ET_norm', 'TBV_norm', 'LHFR_norm']
    ]
    # mode: "HvE" (Healthy vs Eczema); "severity" (binary) <<<<<<<
    for mode in ['HvE', 'severity']:
        results_dict = {}
        test_all_acc = {}
        test_all_cmat = {}
        test_all_results = {}
        for clf_choice in clf_list:
            for feats_choice in feats_strat:
                # load test set, same for all
                df_test = pd.read_csv('../data/features_norm_test.csv')
                # make run name
                run_name = '{}|{}|'.format(mode, clf_choice)
                run_name += '-'.join(feats_choice)
                # initialize results list
                results_train = []
                results_val = []
                results_test = []
                results_to_save = []
                test_acc = []
                test_cmat = []
                # cv loop
                for cv_idx in range(6):
                    if mode == 'severity':
                        # load data
                        data_fname = '../data/CV{}/coord_{}.csv'
                        df_train = pd.read_csv(data_fname.format(cv_idx, 'train'))
                        df_val = pd.read_csv(data_fname.format(cv_idx, 'valid'))
                        # remove elements with label = 3, i.e. healthy
                        df_train = df_train[df_train['label'] != 3]
                        df_val = df_val[df_val['label'] != 3]
                        df_test = df_test[df_test['label'] != 3]
                        # group severity labels
                        df_train['exp_label'] = df_train['label'].apply(group_severity)
                        df_val['exp_label'] = df_val['label'].apply(group_severity)
                        df_test['exp_label'] = df_test['label'].apply(group_severity)
                    elif mode == 'HvE':
                        # load data
                        data_fname = '../data/healthy_v_eczema/CV{}/coord_{}.csv'
                        df_train = pd.read_csv(data_fname.format(cv_idx, 'train'))
                        df_val = pd.read_csv(data_fname.format(cv_idx, 'valid'))
                        # change labels
                        df_train['exp_label'] = df_train['label'].apply(group_healthy_v_ecz)
                        df_val['exp_label'] = df_val['label'].apply(group_healthy_v_ecz)
                        df_test['exp_label'] = df_test['label'].apply(group_healthy_v_ecz)
                    else:
                        raise Exception("Select mode.")
                    # take features and labels as np array
                    feats_train = df_train[feats_choice].values
                    labels_train = df_train['exp_label'].values
                    feats_val = df_val[feats_choice].values
                    labels_val = df_val['exp_label'].values
                    feats_test = df_test[feats_choice].values
                    labels_test = df_test['exp_label'].values
                    # init classifier
                    # C: regularization parameter
                    if clf_choice == 'SVM':
                        clf = SVC(kernel='linear', C=1., probability=False, class_weight='balanced')
                    elif clf_choice == 'RF':
                        random_forest_kwargs = {
                            'n_estimators': 25, 'criterion': 'entropy', 'max_depth': 5, 'min_samples_split': 2,
                            'min_samples_leaf': 1,
                            'min_weight_fraction_leaf': 0.0, 'max_features': 1, 'max_leaf_nodes': None,
                            'min_impurity_decrease': 0.0,
                            'min_impurity_split': None, 'bootstrap': True, 'oob_score': False, 'n_jobs': None,
                            'random_state': 7487, 'verbose': 0, 'warm_start': False,
                            'class_weight': 'balanced', 'ccp_alpha': 0.0, 'max_samples': None
                        }
                        clf = RandomForestClassifier(**random_forest_kwargs)
                    else:
                        raise Exception("no classifier selected")
                    clf.fit(feats_train, labels_train)
                    # train_pred_prob = clf.decision_function(feats_train)
                    # results
                    results_tmp = evaluate_results(clf, feats_train, labels_train)
                    results_train.append(results_tmp)
                    results_tmp = evaluate_results(clf, feats_val, labels_val)
                    results_val.append(results_tmp)
                    results_to_save.append(results_tmp['acc'])

                # append to results dict
                results_dict[run_name] = results_to_save

        # make df
        results_df = pd.DataFrame(results_dict).transpose()
        results_mean = results_df.mean(axis=1)
        results_mean.rename('mean', inplace=True)
        results_std =  results_df.std(axis=1)
        results_std.rename('std', inplace=True)
        results_cmpt = results_df.join(results_mean)
        results_cmpt = results_cmpt.join(results_std)
        results_cmpt.to_csv('../saves/{}_res_cmpt.csv'.format(mode))
        print('validation results'.ljust(20, '<'))
        for idx, row in results_cmpt.iterrows():
            print('{}: {:.2f}\\pm{:.2f}'.format(idx, row['mean'], row['std']))
