

class MildVsModSev:
    def __init__(self, datasets, no_pt_data, train_valid_test):
        self.name = 'mild_vs_modsev'
        self.datasets = datasets
        self.no_pt_data = no_pt_data
        self.train_valid_test = train_valid_test

        self.c_label_names = ['mild', 'moderate', 'severe', 'healthy']
        self.labels = [0, 1, 1, 1000]
        self.tgt_cls_count_train = [450, 225, 225, 0]
        self.tgt_cls_count_valid = [45, 23, 23, 0]
        self.tgt_cls_count_test  = [30, 15, 15, 0]
        self.n_time_cropping = None
        self.n_time_cropping_LFHF = None

        """prepare_data_mild_vs_modsev."""
        if self.datasets['mild'] is None or self.datasets['moderate'] is None or \
                self.datasets['severe'] is None or self.datasets['healthy'] is None:
            raise ValueError('Exist None data in datasets. Please run prepare_train_valid_data first.')

