import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from tensorflow import keras
import tensorflow as tf


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
        # print(y, end=' ')
        y = [data[1] + 2 * data[2] + 3 * data[3] for data in y]  # convert data from the form [_, r, l, lr] to [0, 1, 2, or 3]
        # print('becomes', y)
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
        # preds = self.clf.predict_proba(x)
        # print(preds, y)
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
        if 'model' in kwargs:
            self.model_location = kwargs['model']
        else:
            self.model_location = 'model'
        try:
            self.clf = keras.models.load_model(self.model_location)
            print('loaded {}!'.format(self.model_location))
            for i in range(len(self.clf.layers[:-1])):
                print(self.clf.layers[i])
                self.clf.layers[i].trainable = False
            self.clf.summary()
        except:
            self.clf = keras.models.Sequential()
            self.clf.add(keras.layers.Dense(100, input_dim=800, activation="relu"))
            self.clf.add(keras.layers.Dense(50, activation="relu"))
            self.clf.add(keras.layers.Dense(20, activation="relu"))
            self.clf.add(keras.layers.Dense(4, activation="softmax"))
            self.clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.clf.summary()
            self.clf.save(self.model_location)

    def fit(self, x, y, epochs=100):
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.clf.fit(x, y, epochs=epochs, validation_split=0.2, callbacks=[es])
        self.clf.save('model')
        return self.clf

    def predict(self, x):
        if isinstance(x[0], list):
            preds = self.clf.predict(x)
        else:
            preds = self.clf.predict([x])
        out = [[1 if i == max(pred) else 0 for i in pred] for pred in preds]
        return out


class RNN(Classifier):
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model_location = kwargs['model']
        else:
            self.model_location = 'rnn_model'
        try:
            self.clf = keras.models.load_model(self.model_location)
            print('loaded {}!'.format(self.model_location))
            for i in range(len(self.clf.layers[:-1])):
                print(self.clf.layers[i])
                self.clf.layers[i].trainable = False
            self.clf.summary()
        except:
            inputs = keras.Input(shape=(800,))

            expand_dims = tf.expand_dims(inputs, axis=2)
            gru = tf.keras.layers.GRU(32, return_sequences=True)(expand_dims)
            flatten = tf.keras.layers.Flatten()(gru)
            outputs = tf.keras.layers.Dense(4, activation='softmax')(flatten)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.clf = model
            self.clf.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            self.clf.summary()

    def fit(self, x, y, epochs=100):
        print('xshape =', np.array(x).shape)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.clf.fit(x, y, epochs=epochs, validation_split=0.2, batch_size=32, callbacks=[es])
        self.clf.save(self.model_location)
        return self.clf

    def predict(self, x):
        if isinstance(x[0], list):
            preds = self.clf.predict(x)
        else:
            preds = self.clf.predict([x])
        out = [[1 if i == max(pred) else 0 for i in pred] for pred in preds]
        return out


class CNN(Classifier):
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model_location = kwargs['model']
        else:
            self.model_location = 'cnn_model'
        try:
            self.clf = keras.models.load_model(self.model_location)
            print('loaded {}!'.format(self.model_location))
            for i in range(len(self.clf.layers[:-1])):
                self.clf.layers[i].trainable = False
            self.clf.summary()
        except:
            self.clf = keras.models.Sequential()
            self.clf.add(keras.layers.Conv1D(filters=50, kernel_size=4, padding='same', activation='relu', input_shape=(100, 8)))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Conv1D(50, 4, activation="relu"))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Conv1D(50, 4, activation="relu"))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Flatten())
            self.clf.add(keras.layers.Dense(1000, activation="softmax"))
            self.clf.add(keras.layers.Dense(4, activation="softmax"))
            self.clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.clf.summary()
            self.clf.save(self.model_location)
        tf.keras.utils.plot_model(
            self.clf,
            to_file="model.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
        )

    def fit(self, x, y, epochs=100):
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        es2 = keras.callbacks.EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=10)
        print(np.array(x).shape, np.array(y).shape)
        x = np.array(x)
        x = x.reshape((x.shape[0], 100, 8))
        print(np.array(x).shape, np.array(y).shape)
        self.clf.fit(x, y, epochs=epochs, validation_split=0.2, callbacks=[es, es2])
        # self.clf.save('model')
        return self.clf

    def predict(self, x, min_probability=None):
        if isinstance(x[0], list):
            x = np.array(x).flatten()
        else:
            x = np.array([x]).flatten()
        x = x.reshape((int(len(x)/800), 100, 8))
        preds = self.clf.predict(x)
        # for item in preds:
        #     for pred in item:
        #         print('{:.3f} '.format(pred), end='')
        #     print()
        out = []
        for item in preds:
            if min_probability is not None and max(item) < min_probability:
                out += [[1, 0, 0, 0]]
            else:
                out += [[1 if i == max(item) else 0 for i in item]]
        return out


class CNN2(CNN):
    def __init__(self, **kwargs):
        if 'model' in kwargs:
            self.model_location = kwargs['model']
        else:
            self.model_location = 'cnn_model2'
        try:
            self.clf = keras.models.load_model(self.model_location)
            print('loaded {}!'.format(self.model_location))
            for i in range(len(self.clf.layers[:-1])):
                self.clf.layers[i].trainable = False
            self.clf.summary()
        except:
            self.clf = keras.models.Sequential()
            self.clf.add(keras.layers.Conv1D(filters=50, kernel_size=4, padding='same', activation='sigmoid', input_shape=(100, 8)))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Conv1D(50, 4, activation="sigmoid"))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Conv1D(50, 4, activation="sigmoid"))
            self.clf.add(keras.layers.BatchNormalization())
            self.clf.add(keras.layers.MaxPooling1D(2))
            self.clf.add(keras.layers.Flatten())
            self.clf.add(keras.layers.Dense(100, activation="sigmoid"))
            self.clf.add(keras.layers.Dense(4, activation="softmax"))
            self.clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.clf.summary()
            self.clf.save(self.model_location)
        tf.keras.utils.plot_model(
            self.clf,
            to_file="model2.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
            layer_range=None,
        )


if __name__ == "__main__":
    LDA().fit([], [[0, 0], [0, 1], [1, 0], [1, 1]])
    data = np.array([[x for x in range(800)] for _ in range(4)])
    y = np.array([[0, 0, 1, 0] for _ in range(4)])
    # print(data)
    # print(y)
    # ANN()
    CNN2()#.fit(data, y)
    # preds = [[0.1, 0.1, 0.9, 0.1], [0.2, 0.3, 0.4, 0.9]]
    # out = [[1 if i == max(pred) else 0 for i in pred] for pred in preds]
    # print(preds)
    # print(out)
