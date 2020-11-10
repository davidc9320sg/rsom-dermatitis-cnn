from DataPreprocessing.DataLoader import DataLoader


if __name__ == '__main__':
    # experiment_type = ['healthy_vs_disease','mild_vs_modsev', 'three_severities']
    experiment_type = ['mild_vs_modsev']


    for expt_type in experiment_type:
        data_loader = DataLoader(expt_type)
        data_loader.generate_test_data()
        quit()
        for cv in [5]: #range(1, 2):
            print('###################################################################################')
            print('experiment type: {} cv:{}/5'.format(expt_type, cv))
            dataloader.generate_train_valid_data(cv)
            print('####################################################################################')


