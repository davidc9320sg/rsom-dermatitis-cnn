

class ThreeSeverities:
    def __init__(self, datasets, no_pt_data, train_valid_test):
        self.name = 'three_severities'
        self.datasets = datasets
        self.no_pt_data = no_pt_data
        self.train_valid_test = train_valid_test

        self.c_label_names = ['mild', 'moderate', 'severe', 'healthy']
        self.labels = [0, 1, 2, 1000]
        self.tgt_cls_count_train = [400, 400, 400, 0]
        self.tgt_cls_count_valid = [40, 40, 40, 0]
        self.tgt_cls_count_test  = [30, 30, 30, 0]
        self.n_time_cropping = None
        self.n_time_cropping_LFHF = None

        """prepare_data_mild_vs_modsev."""
        if self.datasets['mild'] is None or self.datasets['moderate'] is None or \
                self.datasets['severe'] is None or self.datasets['healthy'] is None:
            raise ValueError('Exist None data in datasets. Please run prepare_train_valid_data first.')
