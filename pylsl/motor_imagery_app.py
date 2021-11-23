"""Example program to show how to read a multi-channel time series from LSL."""
import math
import threading

# import pygame
from random import random

from sklearn.preprocessing import OneHotEncoder

from pylsl import StreamInlet, resolve_stream
import numpy as np
import pandas as pd
import time
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
from statistics import mode
from datetime import datetime
import sys
import os
import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('error')


def handle_keyboard_chunk(chunk, keys):
    ''' Returns the button statuses from the LSL keyboard chunk '''
    ks, times = chunk
    new_chunk = [[], []]
    for i in range(len(ks)):
        if ks[i][0] in ('LCONTROL pressed', 'LCONTROL released', 'RCONTROL pressed', 'RCONTROL released'):
            new_chunk[0].append(ks[i])
            new_chunk[1].append(times[i])
    chunk = tuple(new_chunk)
    if not chunk[0]:
        if keys is None:
            return [[0, 0, 0]], False

        return keys, False

    if keys is None:
        keys = [[0, 0, 0]]
    else:
        keys = list(keys[-1][:2])

    out = np.zeros((0, 3))  # data should be appended in the format LSHIFT, RSHIFT, TIME
    for i in range(len(chunk[0])):
        action = chunk[0][i][0]
        timestamp = chunk[1][i]
        if action == 'LCONTROL pressed':
            keys[0] = 1
        elif action == 'LCONTROL released':
            keys[0] = 0
        elif action == 'RCONTROL pressed':
            keys[1] = 1
        elif action == 'RCONTROL released':
            keys[1] = 0
        else:
            continue
        out = np.append(out, [keys + [timestamp]], axis=0)

    if len(out) == 0:
        return keys, False

    return out, True


def normalise_list(x):
    x = np.array(x)
    try:
        out = ((x -x.min()) / (x.max() - x.min())).tolist()
    except Warning:
        out = [np.zeros(len(x[0])).tolist()]
    return out


def normalise_eeg(eeg):
    return [normalise_list(eeg[i::8]) for i in range(8)]


def my_filter(x, y, a=None, b=None):
    # b = [0.9174, -0.7961, 0.9174]
    # a = [-1, 0.7961, -0.8347]
    # Parameters for a 40-hz low-pass filter
    if a is None:
        a = [-1, 0.331]
    if b is None:
        b = [0.3345, 0.3345]
    if len(y) > len(a):
        for col in range(len(y[-1][:-3])):
            y[-1][col] = sum(a[i]*y[-1-i][col] + b[i]*x[-1-i][col] for i in range(len(a)))
            # for i in range(len(a)):
            #     y[-1][col] += a[i]*y[-1-i][col] + b[i]*x[-1-i][col]
    return y


def fir_filter(x, y, a=None):
    if a is None:
        a = [1.4, -0.8, 1.4]  # 50 Hz notch filter
        # a = [1]  # do nothing
    if len(x) >= len(a):
        for col in range(len(y[-1][:-3])):
            y[-1][col] = sum([a[i]*x[-1-i][col] for i in range(len(a))])
            # print(y[-1][col])
    return y


class EEG:
    def __init__(self, user_id, game, data_length=100, ignore_lsl=False, ignore_BCI=False):
        # first resolve an EEG stream on the lab network

        self.user_id = user_id
        self.game = game
        self.data_length = data_length

        if not ignore_lsl:
            print("looking for an Keyboard stream...")
            self.keyboard = resolve_stream('name', 'Keyboard')
            print(self.keyboard)
            self.keyboard_inlet = StreamInlet(self.keyboard[0])
        if not ignore_lsl and not ignore_BCI:
            print("looking for an EEG stream...")
            self.eeg = resolve_stream('type', 'EEG')
            print(self.eeg)
            self.eeg_inlet = StreamInlet(self.eeg[0])

        self.eeg_dataset = []  # of the format [channel0, c1, ..., timestamp, left_shift, right_shift]
        self.filtered = []
        self.fft = []
        self.keys = None

        self.running = False
        self.clf = None

    @property
    def prev_dl(self):
        return np.array([item[:-3] for item in self.filtered[-1:-1-self.data_length:-1]]).T.tolist()

    def eeg_sample(self, data=None):
        if data is None:
            sample, timestamp = self.eeg_inlet.pull_sample()
            data = [sample + [timestamp] + list(self.keys[-1][:2])]

        self.eeg_dataset += data
        self.filtered += [[0]*8 + list(data[0][-3:])]
        # self.filtered = my_filter(self.eeg_dataset, self.filtered)
        # self.filtered = my_filter(self.eeg_dataset, self.filtered, a=[-1, 1.452, -0.4523], b=[0.2737, 0, -0.2737])
        self.filtered = my_filter(self.eeg_dataset, self.filtered,
                   b=[float(i) for i in '0.3749   -0.2339         0    0.2339   -0.3749'.split()],
                   a=[-1*float(i) for i in '1.0000   -1.8173    1.9290   -1.3011    0.2154'.split()])  # this one also works well!
        # self.filtered = fir_filter(self.eeg_dataset, self.filtered)  # this works well!

        if len(self.filtered) > self.data_length:
            norm = normalise_eeg(self.prev_dl)
            fft = norm#np.array([np.abs(np.fft.fft(n)) for n in norm]).flatten().tolist()
            self.fft += [fft + self.filtered[-1][-2:]]

    def mi_to_fft(self):
        hist_mi = [f for f in os.listdir('users/data') if 'mi_' + self.user_id == f[:5]]
        hist_fft = [f for f in os.listdir('users/data') if 'fft_' + self.user_id == f[:6]]
        needed_hist_fft = []
        for fmi in hist_mi:
            if 'fft_' + fmi[3:] not in hist_fft:
                needed_hist_fft.append(fmi)
        print('need to convert to fft:', needed_hist_fft)

        print('loading {}'.format(needed_hist_fft))
        for mi_file in needed_hist_fft:
            loaded_data = np.load('users/data/' + mi_file)
            self.eeg_dataset = []
            self.filtered = []
            self.fft = []
            t0 = time.time()

            for row in range(len(loaded_data)):
                data = [loaded_data[row]]
                self.eeg_sample(data)

                if row % 1000 == 500:
                    tr = (time.time() - t0) * (len(loaded_data) - row) / row
                    print('time remaining: {}'.format(tr))

            print()
            fft_name = 'users/data/fft_' + mi_file[3:]
            print('outputting to', fft_name)
            np.save(fft_name, self.fft)
            # print(pd.DataFrame(self.fft))
            # good = 'users/data/good_' + mi_file[3:]
            # good = np.load(good)
            # print(pd.DataFrame(good))
            #
            # print(f'{np.array_equal(self.fft, good) = }')

    def gather_data(self):
        thread = threading.Thread(target=self.__gather)
        thread.start()
        return thread

    def __gather(self):
        self.running = True

        self.eeg_dataset = []
        self.filtered = []
        self.fft = []
        while self.running:
            # get a new sample (you can also omit the timestamp part if you're not interested in it)
            chunk = self.keyboard_inlet.pull_chunk()
            self.keys, is_new = handle_keyboard_chunk(chunk, self.keys)

            self.eeg_sample()  # get and process the latest sample from the EEG headset
        self.save_training()

    def train(self, classifier='KNN', include_historical=False, **kwargs):
        thread = threading.Thread(target=self.__train, args=(classifier, include_historical), kwargs=kwargs)
        thread.start()
        return thread

    def __train(self, classifier='KNN', include_historical=False, **kwargs):
        print('data recording complete. building model... (this may take a few moments)')
        # hist_fft = [f for f in os.listdir('users/data') if 'fft_' + self.user_id in f and 'npy' in f]  # grab historical data for user
        #
        # # take only the most recent data if we don't include_historical
        # if not include_historical or classifier == 'ANN':
        #     print('ignoring historical data...')
        #     hist_fft = [hist_fft[-1]]
        #
        # print('loading {}'.format(hist_fft))
        # data = [np.load('users/data/' + f).tolist()[::5] for f in hist_fft]
        #
        # # X = [dat[:][:-2] for dat in data]
        # # Y_i = [dat[:][-2:] for dat in data]
        # # Y_o = []
        # # X_o = []
        # data_o = []
        #
        # # merge historical data together
        # for i in range(len(data)):
        #     # Y_o += Y_i[i]
        #     # X_o += X[i]
        #     print('data', i, 'shape', np.array(data[i]).shape)
        #     data_o += data[i]

        def flatten(t):
            return [item for sublist in t for item in sublist]

        def get_fmi_dl(index, data, length=100):
            np_fmi = np.array(data[index:index + length])
            x = flatten(np_fmi[:, :-3].tolist())
            y = np_fmi[-1, -2:].tolist()
            return [x + y]

        data = self.filtered
        data_o = []
        for line in range(len(data)-100):
            data_o += get_fmi_dl(line, data)
        # data_o = data
        print('balancing data')
        # print(data_o)
        print('data shape:', np.array(data_o).shape)
        fft_df = pd.DataFrame(data_o, columns=['c' + str(i) for i in range(802)])

        fft_df['y'] = fft_df.apply(lambda row: row.c800 + 2 * row.c801, axis=1)
        fft_df = fft_df.loc[fft_df['y'] != 3].reset_index(drop=True)

        m = min(fft_df.y.value_counts())  # grab the count of the least common y value (left, right, or none)
        y_vals = fft_df.y.unique()
        print('got min={}, unique={}'.format(m, y_vals))

        randomized_df = fft_df.sample(frac=1).reset_index(drop=True)
        out = np.zeros((m*3, 803))

        for i, y in enumerate(y_vals):
            arr = randomized_df.loc[randomized_df['y'] == y].head(m).to_numpy()
            out[i*m:i*m + m] = arr
        print('consolidated data')
        randomized_df = pd.DataFrame(out)
        randomized_df = randomized_df.sample(frac=1).reset_index(drop=True)
        print('reordered data')
        Y = randomized_df[[800, 801]].to_numpy()
        del randomized_df[800], randomized_df[801], randomized_df[802]
        X = randomized_df.to_numpy()
        print('created X and Y. X.shape={}, Y.shape={}'.format(X.shape, Y.shape))
        # y =

        # one hot encoding for Y values
        # Y_i = list(Y_o)
        Y_i = [[0], [1], [2], [3]] + [[2*Y[i][-2] + Y[i][-1]] for i in range(len(Y))]
        enc = OneHotEncoder()
        print('fitting one hot encoder')
        enc.fit(Y_i)
        # X = X_o
        Y = enc.transform(Y_i).toarray()[4:]

        if len(X) == 0 or len(Y) == 0:
            print('no training data provided')
            return

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)
        # if classifier == 'KNN':
        #     self.clf = models.KNN(n_neighbors=3, **kwargs)
        # elif classifier == "LDA":
        #     self.clf = models.LDA()
        # elif classifier == "SVM":
        #     self.clf = models.SVM(**kwargs)
        # elif classifier == "ANN":
        #     self.clf = models.ANN(**kwargs)
        # elif classifier == "RNN":
        #     self.clf = models.RNN(**kwargs)
        # elif classifier == "CNN":
        #     self.clf = models.CNN(**kwargs)
        # else:
        #     print('no valid classifier provided ({}). Using KNN'.format(classifier))
        #     self.clf = models.KNN(n_neighbors=3)
        print('training model ({} classifier)...'.format(self.clf))
        self.clf.fit(X_train, Y_train)
        print('analysing model...')
        preds = self.clf.predict(X_test)
        print('combined acc:', accuracy_score(Y_test, preds))
        print()

        print('model complete.')

    def build_model(self, classifier, **kwargs):
        thread = threading.Thread(target=self._build_model, args=(classifier, ), kwargs=kwargs)
        thread.start()
        return thread

    def _build_model(self, classifier, **kwargs):
        if classifier == 'KNN':
            self.clf = models.KNN(n_neighbors=3, **kwargs)
        elif classifier == "LDA":
            self.clf = models.LDA()
        elif classifier == "SVM":
            self.clf = models.SVM(**kwargs)
        elif classifier == "ANN":
            self.clf = models.ANN(**kwargs)
        elif classifier == "RNN":
            self.clf = models.RNN(**kwargs)
        elif classifier == "CNN":
            self.clf = models.CNN2(**kwargs)
        else:
            print(f'no valid classifier provided ({classifier}). Using KNN')
            self.clf = models.KNN(n_neighbors=3)

    def save_training(self):
        suffix = '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        print('saving eeg data:', np.array(self.eeg_dataset).shape)
        eeg_file = './users/data/mi_' + self.user_id + suffix
        np.save(eeg_file, self.eeg_dataset)

        print('saving filtered eeg data:', np.array(self.filtered).shape)
        filt_eeg_file = './users/data/fmi_' + self.user_id + suffix
        np.save(filt_eeg_file, self.filtered)

        print('saving filtered fft data:', np.array(self.fft).shape)
        fft_eeg_file = './users/data/fft_' + self.user_id + suffix
        np.save(fft_eeg_file, self.fft)

    def test(self, send_to=None):
        thread = threading.Thread(target=self.__test, args=(send_to, ))
        thread.start()
        return thread

    def __test(self, send_to=None):
        assert self.clf

        self.running = True
        self.eeg_dataset = []
        self.filtered = []
        self.fft = []
        last_preds = []

        def flatten(t):
            return [item for sublist in t for item in sublist]

        def get_fmi_dl(index, data, length=100):
            np_fmi = np.array(data[index:index + length])
            x = flatten(np_fmi[:, :-3].tolist())
            return [x]

        while self.running:
            self.eeg_sample()
            if len(self.filtered) > self.data_length:
                pred = self.clf.predict(get_fmi_dl(-101, self.filtered))
                # if pred[0][2]:
                #     last_preds += [1]
                # elif pred[0][1]:
                #     last_preds += [-1]
                # else:
                #     last_preds += [0]
                # if len(last_preds) >= 25:
                #     last_preds = last_preds[1:]
                # avg = sum(last_preds) / len(last_preds)
                # left = avg < -0.25
                # right = avg > 0.25
                left = pred[0][0] or pred[0][2]
                right = pred[0][1] or pred[0][2]
                if send_to:
                    send_to((left, right))
            elif send_to:
                send_to((0, 0))

    def close(self):
        print('closing eeg and keyboard streams')
        if hasattr(self, 'eeg_inlet'):
            self.eeg_inlet.close_stream()
        if hasattr(self, 'keyboard_inlet'):
            self.keyboard_inlet.close_stream()


def main(user_id, train_time=30, test_time=30, classifier='LDA'):
    import motor_bci_game

    while len(user_id) != 2:
        user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
        if len(user_id) == 2:
            print('user_id={}'.format(user_id))
            break
        print('user ID must be 2 digits, you put', len(user_id))

    game = motor_bci_game.Game()
    eeg = EEG(user_id, game)
    gathering = eeg.gather_data()  # runs in background
    eeg.build_model(classifier=classifier)#, model_location="cnn_model_8_11_22_32")  # runs in background
    game.run_keyboard(run_time=train_time)  # runs in foreground
    eeg.running = False
    while gathering.is_alive(): pass

    training = eeg.train(classifier=classifier, include_historical=False)#, model_location='cnn_model_8_11_22_32')  #, decision_function_shape="ovo")
    while training.is_alive(): pass
    eeg.running = False  # stop eeg gathering once game completes
    time.sleep(5)

    print('testing')
    testing = eeg.test(send_to=game.p1.handle_keys)
    game.run_eeg(test_time)
    eeg.running = False
    while testing.is_alive():
        pass

    eeg.close()
    print('scores:', game.e.scores)
    game.quit()
    sys.exit()


def main_game_2(user_id, train_time=30, test_time=30, classifier='CNN'):
    import game_2

    while len(user_id) != 2:
        user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
        if len(user_id) == 2:
            print('user_id={}'.format(user_id))
            break
        print('user ID must be 2 digits, you put', len(user_id))

    game = game_2.Game()
    eeg = EEG(user_id, game)
    gathering = eeg.gather_data()  # runs in background
    eeg.build_model(classifier=classifier)#, model_location="cnn_model_8_11_22_32")  # runs in background
    game.run_keyboard(run_time=train_time)  # runs in foreground
    eeg.running = False
    while gathering.is_alive(): pass

    training = eeg.train(classifier=classifier, include_historical=False)#, model_location='cnn_model_8_11_22_32')  #, decision_function_shape="ovo")
    while training.is_alive(): pass
    eeg.running = False  # stop eeg gathering once game completes
    # time.sleep(5)

    print('testing')
    testing = eeg.test(send_to=game.block.handle_keys)
    game.run_eeg(test_time)
    eeg.running = False
    while testing.is_alive():
        pass

    eeg.close()
    print('scores:', game.block.scores)
    game.quit()
    sys.exit()


def train_test(user_id):
    import motor_bci_game

    # while len(user_id) != 2:
    #     user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
    #     if len(user_id) == 2:
    #         print(f'{user_id=}')
    #         break
    #     print('user ID must be 2 digits, you put', len(user_id))

    game = motor_bci_game.Game()
    eeg = EEG(user_id, game, ignore_lsl=True)

    training = eeg.train(classifier='CNN', include_historical=False, model='new_test')  #, decision_function_shape="ovo")
    while training.is_alive(): pass

    eeg.close()
    print('scores:', game.e.scores)
    game.quit()
    sys.exit()


def convert_mi_to_fft(user_id):
    import motor_bci_game

    # user_id = '00'# + str(i)
    print('user_id={}'.format(user_id))
    game = motor_bci_game.Game()
    eeg = EEG(user_id, game, ignore_lsl=True)
    eeg.mi_to_fft()

    eeg.close()
    game.quit()
    sys.exit()


if __name__ == '__main__':
    user_id = '-1'
    mode = 5
    if mode == 1:
        good = np.load('users/data/fmi_01_300921_211231.npy')
        print(pd.DataFrame(good))

    elif mode == 2:
        main(user_id=user_id,
             train_time=10,
             test_time=60)

    elif mode == 3:
        convert_mi_to_fft(user_id)

    elif mode == 4:
        train_test(user_id)

    elif mode == 5:
        main_game_2(user_id=user_id,
                    train_time=30,
                    test_time=30)

    print('done?')
