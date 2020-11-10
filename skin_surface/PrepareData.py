import csv


class PrepareData:

    def __init__(self):

        self.data = 'dataset_real.csv'

        self.study, self.labels, self.dirs = self.readMyFile(self.data)
        self.n_mild, self.n_moderate, self.n_severe = self.c_num(self.labels)

        self.mild, self.moderate, self.severe = self.sort_split(self.study, self.labels, self.dirs,
                                                                self.n_mild, self.n_moderate, self.n_severe)

    def readMyFile(self, data):
        study = list()
        labels = list()
        dirs = list()

        with open(data) as csvDataFile:
            csvReader = csv.reader(csvDataFile, delimiter=",")
            for row in csvReader:
                study.append(row[0])
                labels.append(row[1])
                dirs.append(row[2])

        # print(study,labels,dirs)

        return study, labels, dirs

    def c_num(self, labels):

        n_mild =0
        n_moderate=0
        n_severe=0
        for i in labels:
            if i == '0':
                n_mild += 1
            elif i == '1':
                n_moderate += 1
            else :
                n_severe += 1
        return n_mild, n_moderate, n_severe

    def sort_split(self, study, labels, dirs, n_mild,n_moderate,n_severe):

        sort_index = sorted(range(len(labels)), key=lambda k: labels[k])
        study = [study[i] for i in sort_index]
        labels = [labels[i] for i in sort_index]
        dirs = [dirs[i] for i in sort_index]

        mild = study[:n_mild], labels[:n_mild], dirs[:n_mild]
        moderate = study[n_mild:n_mild + n_moderate], labels[n_mild:n_mild + n_moderate], dirs[
                                                                                          n_mild:n_mild + n_moderate]
        severe = study[n_mild + n_moderate:], labels[n_mild + n_moderate:], dirs[n_mild + n_moderate:]

        return mild, moderate, severe




