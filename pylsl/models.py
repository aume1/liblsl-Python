import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tensorflow import keras


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
        if len(np.array(x).shape) == 0:
            x = [x]
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
        if np.array(x).ndim == 1:
            x = [x]
        # print(f'{np.array(x).ndim=}')
        # print(f'{x=}')
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


class ANN(Classifier):
    def __init__(self, **kwargs):
        self.clf = keras.models.Sequential()
        self.clf.add(keras.layers.Dense(100, input_dim=800, activation="relu"))
        # self.clf.add(keras.layers.Dense(100, activation="relu"))
        self.clf.add(keras.layers.Dense(4, activation="softmax"))
        self.clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x, y, epochs=30):
        self.clf.fit(x, y, epochs=epochs)
        return self.clf

    def predict(self, x):
        preds = self.clf.predict(x)
        out = [[1 if i == max(pred) else 0 for i in pred] for pred in preds]
        return out


if __name__ == "__main__":
    preds = [[0.1, 0.1, 0.9, 0.1], [0.2, 0.3, 0.4, 0.9]]
    out = [[1 if i == max(pred) else 0 for i in pred] for pred in preds]
    print(preds)
    print(out)
