

class HealthyVsDisease:
    def __init__(self, datasets, no_pt_data, train_valid_test):
        self.name = 'healthy_vs_disease'
        self.datasets = datasets
        self.no_pt_data = no_pt_data
        self.train_valid_test = train_valid_test

        self.c_label_names = ['mild', 'moderate', 'severe', 'healthy']  #CANNOT CHANGE THE SEQUENCE
        self.labels = [0, 0, 0, 1]  #CANNOT CHANGE THE SEQUENCE
        self.tgt_cls_count_train = [150, 150, 150, 450]  #CANNOT CHANGE THE SEQUENCE
        self.tgt_cls_count_valid = [15, 15, 15, 45]  #CANNOT CHANGE THE SEQUENCE
        self.tgt_cls_count_test  = [10, 10, 10, 30]
        self.n_time_cropping = None
        self.n_time_cropping_LFHF = None

        """check if datasets has None"""
        if self.datasets['mild'] is None or self.datasets['moderate'] is None or \
                self.datasets['severe'] is None or self.datasets['healthy'] is None:
            raise ValueError('Exist None data in datasets. Please run prepare_train_valid_data first.')

