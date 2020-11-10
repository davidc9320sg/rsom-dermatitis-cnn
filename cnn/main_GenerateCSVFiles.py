"""This is a code to generate six different shuffles of train and valid data """

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold


def get_class_count(df, cv, train_valid):
    output_df = pd.DataFrame(columns=['cv', 'train/valid', 'label', 'no_pt'])
    labels = [0, 1, 2, 3]
    for i, label in enumerate(labels):
        output_df.loc[i] = [cv, train_valid, label, len(df[df['label'] == label])]
    return output_df


if __name__ == '__main__':
    test_data = pd.read_csv('../data/features_norm_test.csv')
    test_outputfolder = '../data/test/'
    os.mkdir(test_outputfolder)
    test_data.to_csv(test_outputfolder + 'coord_test.csv', index=False)
    test_output_df = get_class_count(test_data, 'NA', 'test')
    test_output_df.to_csv(test_outputfolder + 'datasets_description.csv', index=False)
    quit()
  

    ### generating training and validation data for CV  ###
    kfold = 6  # number of cross validation

    train = pd.read_csv('../data/features_norm_train.csv')
    valid = pd.read_csv('../data/features_norm_valid.csv')
    data = pd.concat([train, valid], axis=0)

    data = data.sample(frac=1)
    X = data.drop(['label'], axis=1)
    y = data['label']

    kfold = StratifiedKFold(kfold, shuffle=True, random_state=42)
    for i, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid, y_valid = X.iloc[valid_index], y.iloc[valid_index]
        X_train.insert(1, y_train.name, y_train)
        X_valid.insert(1, y_valid.name, y_valid)
        train_data, valid_data = X_train, X_valid

        """writing coord_train and coord_valid.csv """
        outputfolder = '../data/CV{}/'.format(i)
        os.mkdir(outputfolder)
        train_data.to_csv(outputfolder + 'coord_train.csv', index=False)
        valid_data.to_csv(outputfolder + 'coord_valid.csv', index=False)

        train_output_df = get_class_count(train_data, i, 'train')
        valid_output_df = get_class_count(valid_data, i, 'valid')
        output_df = pd.concat([train_output_df, valid_output_df], axis=0)

        output_df.to_csv(outputfolder + 'dataset_description.csv', index=False)



