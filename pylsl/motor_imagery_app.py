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

warnings.filterwarnings('error')


def handle_keyboard_chunk(chunk: tuple, keys):
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
        out = list((x -x.min()) / (x.max() - x.min()))
    except Warning:
        out = list(np.zeros(len(x)))
    return out


def normalise_eeg(eeg):
    return np.array([normalise_list(eeg[i::8]) for i in range(8)]).flatten()


def threaded_analyse(eeg, model):
    norm = normalise_eeg(eeg)
    X_fft = [np.fft.fft(norm).real]
    pred = model.predict(X_fft)
    print(pred, end='')
    if pred[0][0] == 1:
        print('right')
    elif pred[0][1] == 1:
        print('left')
    elif pred[0][2] == 1:
        print()
    elif pred[0][3] == 1:
        print('both')


def my_filter(x, y):
    a = [0.9174, -0.7961, 0.9174]
    b = [1, -0.7961, 0.8347]
    if len(y) > len(a):
        for col in range(len(y[-1][:-3])):
            y[-1][col] = b[0]*x[-1][col]
            for i in range(1, len(a)):
                y[-1][col] += a[i]*y[-1-i][col] - b[i]*x[-1-i][col]
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

        self.eeg_dataset = np.zeros((0, 11))  # of the format [channel0, c1, ..., timestamp, left_shift, right_shift]
        self.filtered = np.zeros((0, 11))
        self.fft = np.zeros((0, 8*self.data_length+2))
        self.keys = None

        self.running = False
        self.clf = None

    def fmi_to_fft(self):
        hist_fmi = [f for f in os.listdir('users/data') if 'fmi_' + self.user_id in f]
        hist_fft = [f for f in os.listdir('users/data') if 'fft_' + self.user_id in f]
        needed_hist_fft = []
        for fmi in hist_fmi:
            if 'fft_' + fmi[4:] not in hist_fft:
                needed_hist_fft.append(fmi)
        # print('need to convert to fft:', needed_hist_fft)

        # hist_filt = [f for f in os.listdir('users/data') if 'fmi_' + self.user_id in f]
        print(f'loading {needed_hist_fft}')
        for fmi_file in needed_hist_fft:
            data = np.load('users/data/' + fmi_file)
            fft = np.zeros((len(data) - self.data_length + 1, 8*self.data_length+2))
            print('converting file. this may take a few moments...', end='')
            prev_dl = np.array([data[i][:-3] for i in range(self.data_length)]).flatten()
            print(len(data) - self.data_length)
            for i, row in enumerate(range(self.data_length - 1, len(data))):
                prev_dl = np.append(data[row][:-3], np.roll(prev_dl, 8)[8:])
                norm = normalise_eeg(prev_dl)
                add = np.append(np.fft.fft(norm).real, [data[row][-2:]])
                # add = list(np.fft.fft(norm).real) + list(data[row][-2:])#, axis=1)
                # fft = np.append(fft, [add], axis=0)
                fft[i] = add
                if row % 1000 == 0:
                    print(end='.')
            print()
            fft_name = 'users/data/fft_' + fmi_file[4:]
            print('outputting to', fft_name)
            np.save(fft_name, fft)
            print(pd.DataFrame(fft))
            good = 'users/data/good_' + fmi_file[4:]
            good = np.load(good)
            print(pd.DataFrame(good))

    def preprocess(self):
        self.filtered = np.append(self.filtered, self.eeg_dataset[-1], axis=0)
        self.filtered = my_filter(self.eeg_dataset, self.filtered)

        if len(self.filtered) > self.data_length:
            prev_dl = np.array([self.filtered[-i][:-3] for i in range(self.data_length, 0, -1)]).flatten()
            norm = normalise_eeg(prev_dl)
            add = np.append(np.fft.fft(norm).real, self.filtered[-1][-2:])

            self.fft = np.append(self.fft, [add], axis=0)
        return self.fft

    def gather_data(self):
        thread = threading.Thread(target=self.__gather)
        thread.start()
        return thread

    def __gather(self):
        self.running = True
        # count = 0
        prev_dl = [0]*self.data_length
        while self.running:
            # get a new sample (you can also omit the timestamp part if you're not interested in it)
            chunk = self.keyboard_inlet.pull_chunk()
            self.keys, is_new = handle_keyboard_chunk(chunk, self.keys)
            if hasattr(self, 'eeg_inlet'):
                sample, timestamp = self.eeg_inlet.pull_sample()
            else:
                timestamp = time.time()
                sample = [math.sin(timestamp * i) + random() for i in range(1, 9)]  # generate fake eeg data

            # print(self.keys)
            data = [sample + [timestamp] + list(self.keys[-1][:2])]
            self.eeg_dataset = np.append(self.eeg_dataset, data, axis=0)
            # self.preprocess()
            self.filtered = np.append(self.eeg_dataset, data, axis=0)
            self.filtered = my_filter(self.eeg_dataset, self.filtered)
            if len(self.filtered) == self.data_length:
                prev_dl = np.array([self.filtered[i][:-3] for i in range(self.data_length)]).flatten()
            if len(self.filtered) > self.data_length:
                prev_dl = np.roll(prev_dl, 8)
                prev_dl[:8] = self.filtered[-1][:-3]
                norm = normalise_eeg(prev_dl)
                add = np.append(np.fft.fft(norm).real, self.filtered[-1][-2:])
                self.fft = np.append(self.fft, [add], axis=0)

            # prev_dl = np.roll(prev_dl, 8)
            # prev_dl[:8] = self.filtered[-1][:-3]
            # if len(self.eeg_dataset) >= self.data_length:
            #     prev_dl = np.array([self.eeg_dataset[i][:-3] for i in range(self.data_length)]).flatten()
            # if len(self.filtered) > self.data_length:
            #     prev_dl = np.roll(prev_dl, 8)
            #     prev_dl[:8] = self.filtered[-1][:-3]
            #     norm = normalise_eeg(prev_dl)
            #     add = np.append(np.fft.fft(norm).real, self.filtered[-1][-2:])
            #     # print(add)
            #     self.fft = np.append(self.fft, [add], axis=0)
            # count += 1
        self.save_training()

    def train(self, classifier='KNN', include_historical=False, **kwargs):
        thread = threading.Thread(target=self.__train, args=(classifier, include_historical), kwargs=kwargs)
        thread.start()
        return thread

    def __train(self, classifier='KNN', include_historical=False, **kwargs):
        print('data recording complete. building model... (this may take a few moments)')
        hist_fft = [f for f in os.listdir('users/data') if 'fft_' + self.user_id in f]  # grab historical data for user

        # take only the most recent data if we don't include_historical
        if not include_historical:
            print('ignoring historical data...')
            hist_fft = [hist_fft[-1]]

        print(f'loading {hist_fft}')
        data = [np.load('users/data/' + f) for f in hist_fft]

        X = [dat[:, :-2] for dat in data]
        Y_i = [dat[:, -2:] for dat in data]
        Y_o = np.zeros((0, 2))
        X_o = np.zeros((0, self.data_length*8))
        data_o = np.zeros((0, self.data_length*8 + 2))

        # merge historical data together
        for i in range(len(data)):
            Y_o = np.append(Y_o, Y_i[i], axis=0)
            X_o = np.append(X_o, X[i], axis=0)
            data_o = np.append(data_o, data[i], axis=0)



        # TODO: Balance the dataset
        print('balancing data')
        # print(data_o)
        fft_df = pd.DataFrame(data_o, columns=['c' + str(i) for i in range(802)])

        fft_df['y'] = fft_df.apply(lambda row: row.c800 + 2 * row.c801, axis=1)

        m = min(fft_df.y.value_counts())  # grab the count of the least common y value (left, right, or none)
        y_vals = fft_df.y.unique()
        print(f'got {m=}, unique={y_vals}')

        randomized_df = fft_df.sample(frac=1).reset_index(drop=True)
        out = np.zeros((0, 803))

        for y in y_vals:
            arr = randomized_df.loc[randomized_df['y'] == y].head(m).to_numpy()
            out = np.append(out, [i for i in arr], axis=0)
        print('consolidated data')
        randomized_df = pd.DataFrame(out)
        randomized_df = randomized_df.sample(frac=1).reset_index(drop=True)
        print('reordered data')
        Y = randomized_df[[800, 801]].to_numpy()
        del randomized_df[800], randomized_df[801], randomized_df[802]
        X = randomized_df.to_numpy()
        print(f'created X and Y. {X.shape=}, {Y.shape=}')
        # y =

        # one hot encoding for Y values
        # Y_i = list(Y_o)
        Y_i = [[0], [1], [2], [3]] + [[2*Y[i][-2] + Y[i][-1]] for i in range(len(Y))]
        enc = OneHotEncoder()
        enc.fit(Y_i)
        # X = X_o
        Y = enc.transform(Y_i).toarray()[4:]

        if len(X) == 0 or len(Y) == 0:
            print('no training data provided')
            return

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)
        if classifier == 'KNN':
            self.clf = models.KNN(n_neighbors=3, **kwargs)
        elif classifier == "LDA":
            self.clf = models.LDA()
        elif classifier == "SVM":
            self.clf = models.SVM(**kwargs)
        else:
            print(f'no valid classifier provided ({classifier}). Using KNN')
            self.clf = models.KNN(n_neighbors=3)
        print(f'training model ({self.clf} classifier)...')
        self.clf.fit(X_train, Y_train)
        print('analysing model...')
        preds = self.clf.predict(X_test)
        print('combined acc:', accuracy_score(Y_test, preds))
        print()

        print('model complete.')

    def save_training(self):
        suffix = '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        print('saving eeg data:', len(self.eeg_dataset))
        eeg_file = './users/data/mi_' + self.user_id + suffix
        np.save(eeg_file, self.eeg_dataset)

        print('saving filtered eeg data:', len(self.filtered[:-1]))
        filt_eeg_file = './users/data/fmi_' + self.user_id + suffix
        np.save(filt_eeg_file, self.filtered[:-1])

        print('saving filtered fft data:', len(self.fft))
        fft_eeg_file = './users/data/fft_' + self.user_id + suffix
        np.save(fft_eeg_file, self.fft)

    def test(self, send_to=None):
        thread = threading.Thread(target=self.__test, args=(send_to, ))
        thread.start()
        return thread

    def __test(self, send_to=None):
        assert self.clf
        count = 0

        last_preds = np.array(['']*50)
        self.running = True
        eeg_dataset = self.eeg_dataset.copy()
        prev_dl = np.array([self.filtered[i][:-3] for i in range(self.data_length)]).flatten()
        while self.running:
            sample, timestamp = self.eeg_inlet.pull_sample()
            data = [sample + [timestamp] + [0, 0]]
            eeg_dataset = np.append(eeg_dataset, data, axis=0)
            filtered = np.append(eeg_dataset, data, axis=0)
            # if count > 2:
            filtered = my_filter(eeg_dataset, filtered)

            prev_dl = np.roll(prev_dl, 8)
            prev_dl[:8] = filtered[-1][:-3]
            if count > self.data_length and count % 50 == 0:
                norm = normalise_eeg(prev_dl)
                X_fft = [np.fft.fft(norm).real]
                pred = self.clf.predict(X_fft)

                last_preds = np.roll(last_preds, 1)

                # last_preds[0] = pred
                # pred = mode(last_preds)
                left = pred[0][0] or pred[0][2]
                right = pred[0][1] or pred[0][2]

                if send_to:
                    send_to((left, right))

            count += 1

    def close(self):
        print('closing eeg and keyboard streams')
        if hasattr(self, 'eeg_inlet'):
            self.eeg_inlet.close_stream()
        if hasattr(self, 'keyboard_inlet'):
            self.keyboard_inlet.close_stream()


def main(user_id, train_time=30, test_time=30):
    import motor_bci_game

    while len(user_id) != 2:
        user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
        if len(user_id) == 2:
            print(f'{user_id=}')
            break
        print('user ID must be 2 digits, you put', len(user_id))

    game = motor_bci_game.Game()
    eeg = EEG(user_id, game)#, ignore_BCI=True)
    gathering = eeg.gather_data()  # runs in background
    game.run_keyboard(run_time=train_time)  # runs in foreground
    eeg.running = False  # stop eeg gathering once game completes
    while gathering.is_alive(): pass

    training = eeg.train(classifier='LDA', include_historical=True)  #, decision_function_shape="ovo")
    while training.is_alive(): pass

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


def train_test(user_id):
    import motor_bci_game

    while len(user_id) != 2:
        user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
        if len(user_id) == 2:
            print(f'{user_id=}')
            break
        print('user ID must be 2 digits, you put', len(user_id))

    game = motor_bci_game.Game()
    eeg = EEG(user_id, game, ignore_lsl=True)

    training = eeg.train(classifier='LDA', include_historical=True)  #, decision_function_shape="ovo")
    while training.is_alive(): pass

    eeg.close()
    print('scores:', game.e.scores)
    game.quit()
    sys.exit()


def convert_fmi_to_fft(user_id):
    import motor_bci_game

    # user_id = '00'# + str(i)
    print(f'{user_id=}')
    game = motor_bci_game.Game()
    eeg = EEG(user_id, game, ignore_lsl=True)
    eeg.fmi_to_fft()

    eeg.close()
    game.quit()
    sys.exit()


if __name__ == '__main__':
    mode = 4
    if mode == 1:
        good = np.load('users/data/fmi_01_300921_211231.npy')
        print(pd.DataFrame(good))

    elif mode == 2:
        main(user_id='99',
             train_time=30,
             test_time=30)

    elif mode == 3:
        convert_fmi_to_fft('99')

    elif mode == 4:
        train_test('99')

    print('done?')
