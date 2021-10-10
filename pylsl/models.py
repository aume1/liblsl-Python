import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


class Classifier:
    clf = None
    model = None

    def fit(self, x, y): return
    def predict(self, x): return


class KNN(Classifier):
    def __init__(self, **kwargs):
        self.clf = KNeighborsClassifier(**kwargs)

    def fit(self, x, y):
        self.model = self.clf.fit(x, y)
        return self.model

    def predict(self, x):
        return self.model.predict(x)


class LDA(Classifier):
    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()

    def fit(self, x, y):
        y = [data[1] + 2 * data[2] + 3 * data[3] for data in y]  # convert data from the form [_, r, l, lr] to [0, 1, 2, or 3]
        self.clf.fit(x, y)
        return self.clf

    def predict(self, x):
        if x is None or len(x) == 0:
            return [[0, 0, 0, 0]]
        y = self.clf.predict(x)
        # print(x)
        y_out = np.empty((len(y), 4))
        for i in range(len(y)):
            if y[i] == 0:
                y_out[i] = [1, 0, 0, 0]
            elif y[i] == 1:
                y_out[i] = [0, 1, 0, 0]
            elif y[i] == 2:
                y_out[i] = [0, 0, 1, 0]
            elif y[i] == 3:
                y_out[i] = [0, 0, 0, 1]
            else:
                y_out[i] = [0, 0, 0, 0]
        # print(y_out)
        return y_out


class SVM(Classifier):
    def __init__(self, **kwargs):
        self.clf = svm.SVC(**kwargs)

    def fit(self, x, y):
        y = [data[1] + 2 * data[2] + 3 * data[3] for data in y]
        self.clf.fit(x, y)
        return self.clf

    def predict(self, x):
        if x is None or len(x) == 0:
            return [[0, 0, 0, 0]]
        y = self.clf.predict(x)
        # print(x)
        y_out = np.empty((len(y), 4))
        for i in range(len(y)):
            if y[i] == 0:
                y_out[i] = [1, 0, 0, 0]
            elif y[i] == 1:
                y_out[i] = [0, 1, 0, 0]
            elif y[i] == 2:
                y_out[i] = [0, 0, 1, 0]
            elif y[i] == 3:
                y_out[i] = [0, 0, 0, 1]
            else:
                y_out[i] = [0, 0, 0, 0]
        # print(y_out)
        return y_out