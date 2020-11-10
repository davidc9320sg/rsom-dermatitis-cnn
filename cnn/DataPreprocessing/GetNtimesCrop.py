

class GetNtimesCrop:
    def __init__(self, expt_type):
        self.tgt_cls_count_train = expt_type.tgt_cls_count_train
        self.tgt_cls_count_valid = expt_type.tgt_cls_count_valid
        self.tgt_cls_count_test  = expt_type.tgt_cls_count_test
        self.no_pt_data = expt_type.no_pt_data
        self.name = expt_type.name
        self.train_valid_test = expt_type.train_valid_test

        self.n_time_cropping, self.n_time_cropping_LFHF = self.compute_n_time_cropping()

    def _get_target_class_count(self):
        if self.train_valid_test == 'train':
            return self.tgt_cls_count_train

        if self.train_valid_test == 'valid':
            return self.tgt_cls_count_valid

        if self.train_valid_test == 'test':
            return self.tgt_cls_count_test

    def compute_n_time_cropping(self):
        tgt_ct_mild, tgt_ct_moderate, tgt_ct_severe, tgt_ct_healthy = self._get_target_class_count()

        mild_pt, moderate_pt, severe_pt, healthy_pt = \
        self.no_pt_data[self.no_pt_data['train/valid'] == self.train_valid_test][
            'no_pt'].tolist()

        n_time_cropping = {
            'mild': int(tgt_ct_mild / mild_pt) if mild_pt != 0 else 0,
            'moderate': int(tgt_ct_moderate / moderate_pt) if moderate_pt != 0 else 0,
            'severe': int(tgt_ct_severe / severe_pt) if severe_pt !=0 else 0,
            'healthy': int(tgt_ct_healthy / healthy_pt) if healthy_pt != 0 else 0,
        }
        n_time_cropping_LFHF = {
            'mild': int(tgt_ct_mild / mild_pt) if mild_pt != 0 else 0,
            'moderate': int(tgt_ct_moderate / moderate_pt) if moderate_pt != 0 else 0,
            'severe': int(tgt_ct_severe / severe_pt) if severe_pt != 0 else 0,
            'healthy': int(tgt_ct_healthy / healthy_pt) if healthy_pt !=0 else 0,
        }
        no_count = [n_time_cropping['mild'] * mild_pt,
                    n_time_cropping['moderate'] * moderate_pt,
                    n_time_cropping['severe'] * severe_pt,
                    n_time_cropping['healthy'] * healthy_pt]
        no_count_LFHF = [n_time_cropping_LFHF['mild'] * mild_pt,
                         n_time_cropping_LFHF['moderate'] * moderate_pt,
                         n_time_cropping_LFHF['severe'] * severe_pt,
                         n_time_cropping_LFHF['healthy'] * healthy_pt]

        print('--------------------------------------------------')
        print('experiment type: ', self.name)
        print('no of data for training: ')
        print('Flat data')
        print('mild: ', no_count[0])
        print('moderate: ', no_count[1])
        print('severe: ', no_count[2])
        print('healthy: ', no_count[3])
        print('DISEASE: {} HEALTHY: {}'.format(sum(no_count[:3]), no_count[-1]))
        print('LFHF ')
        print('mild: ', no_count_LFHF[0])
        print('moderate: ', no_count_LFHF[1])
        print('severe: ', no_count_LFHF[2])
        print('healthy: ', no_count_LFHF[3])
        print('DISEASE: {} HEALTHY: {}'.format(sum(no_count_LFHF[:3]), no_count_LFHF[-1]))
        print('--------------------------------------------------')
        return n_time_cropping, n_time_cropping_LFHF

