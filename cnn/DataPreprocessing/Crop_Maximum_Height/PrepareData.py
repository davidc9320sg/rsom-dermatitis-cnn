import csv


class PrepareData:

    def __init__(self, filepath):

        self.study, self.labels, self.dirs, self.lower, self.upper, self.diff, self.feedback = self.readMyFile(filepath)
        self.n_mild, self.n_moderate, self.n_severe, self.n_healthy = self.c_num(self.labels)

        self.mild, self.moderate, self.severe, self.healthy = self.split(self.study, self.labels, self.dirs,self.lower,
                                                                         self.upper, self.diff, self.feedback,
                                                                         self.n_mild, self.n_moderate, self.n_severe)
        self.datasets = {
            'mild': self.mild,
            'moderate': self.moderate,
            'severe': self.severe,
            'healthy': self.healthy
        }

    def readMyFile(self, data):
        study = list()
        labels = list()
        dirs = list()
        lower = list()
        upper = list()
        diff = list()
        feedback = list()

        with open(data) as csvDataFile:
            csvReader = csv.reader(csvDataFile, delimiter=",")
            # csvReader.next()  #python 2.x
            next(csvReader)  # python 3.x
            for row in csvReader:
                study.append(row[0])
                labels.append(row[1])
                dirs.append(row[2])
                lower.append(row[3])
                upper.append(row[4])
                diff.append(row[5])
                feedback.append(row[6])

        # print(study,labels,dirs)

        return study, labels, dirs, lower, upper, diff, feedback

    def c_num(self, labels):

        n_mild =0
        n_moderate=0
        n_severe=0
        n_healthy=0
        for i in labels:
            if i == '0':
                n_mild += 1
            elif i == '1':
                n_moderate += 1
            elif i == '2':
                n_severe += 1
            else:
                n_healthy += 1
        return n_mild, n_moderate, n_severe, n_healthy

    def split(self, study, labels, dirs, lower, upper, diff, feedback, n_mild,n_moderate, n_severe):

        mild = study[:n_mild], labels[:n_mild], dirs[:n_mild], lower[:n_mild], upper[:n_mild], diff[:n_mild], feedback[:n_mild]
        moderate = study[n_mild:n_mild + n_moderate], labels[n_mild:n_mild + n_moderate], dirs[ n_mild:n_mild + n_moderate], lower[n_mild:n_mild + n_moderate], upper[n_mild:n_mild + n_moderate], diff[n_mild:n_mild + n_moderate], feedback[n_mild:n_mild + n_moderate]                                                                                  
        severe = study[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 labels[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 dirs[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 lower[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 upper[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 diff[n_mild + n_moderate:n_mild + n_moderate + n_severe], \
                 feedback[n_mild + n_moderate:n_mild + n_moderate + n_severe]
        healthy = study[n_mild + n_moderate + n_severe:], \
                 labels[n_mild + n_moderate + n_severe:], \
                 dirs[n_mild + n_moderate + n_severe:], \
                 lower[n_mild + n_moderate + n_severe:], \
                 upper[n_mild + n_moderate + n_severe:], \
                 diff[n_mild + n_moderate + n_severe:], \
                 feedback[n_mild + n_moderate + n_severe:]

        return mild, moderate, severe, healthy




