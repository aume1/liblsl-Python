"""Example program to show how to read a multi-channel time series from LSL."""
import threading

# import pygame
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

    out = np.empty((0, 3))  # data should be appended in the format LSHIFT, RSHIFT, TIME
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
    def __init__(self, user_id, game, data_length=100):
        # first resolve an EEG stream on the lab network

        self.user_id = user_id
        self.game = game
        self.data_length = data_length

        print("looking for an EEG stream...")
        self.eeg = resolve_stream('type', 'EEG')
        print(self.eeg)

        print("looking for an Keyboard stream...")
        self.keyboard = resolve_stream('name', 'Keyboard')
        print(self.keyboard)

        # create a new inlet to read from the stream
        self.eeg_inlet = StreamInlet(self.eeg[0])

        self.keyboard_inlet = StreamInlet(self.keyboard[0])
        self.eeg_dataset = np.empty((0, 11))  # of the format [channel0, c1, ..., timestamp, left_shift, right_shift]
        self.filtered = np.empty((0, 11))
        self.fft = np.empty((0, 8*self.data_length+2))
        self.keys = None

        self.running = False
        self.clf = None

    def gather_data(self):
        thread = threading.Thread(target=self.__gather)
        thread.start()
        return thread

    def __gather(self):
        self.running = True
        # count = 0
        # prev_dl = [0]*self.data_length
        while self.running:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            chunk = self.keyboard_inlet.pull_chunk()
            self.keys, is_new = handle_keyboard_chunk(chunk, self.keys)

            sample, timestamp = self.eeg_inlet.pull_sample()

            # print(self.keys)
            data = [sample + [timestamp] + list(self.keys[-1][:2])]
            self.eeg_dataset = np.append(self.eeg_dataset, data, axis=0)
            self.filtered = np.append(self.eeg_dataset, data, axis=0)
            self.filtered = my_filter(self.eeg_dataset, self.filtered)

            # prev_dl = np.roll(prev_dl, 8)
            # prev_dl[:8] = self.filtered[-1][:-3]
            if len(self.filtered) > self.data_length:
                prev_dl = np.array([self.filtered[i][:-3] for i in range(self.data_length)]).flatten()
                norm = normalise_eeg(prev_dl)
                add = np.append(np.fft.fft(norm).real, self.filtered[-1][-2:])
                # print(add)
                self.fft = np.append(self.fft, [add], axis=0)
            # count += 1

    def train(self, classifier='KNN'):
        thread = threading.Thread(target=self.__train, args=(classifier, ))
        thread.start()
        return thread

    def __train(self, classifier='KNN', include_historical=False):
        print('data recording complete. building model... (this may take a few moments)')
        # c = ['Cz', 'Fpz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'time', 'left', 'right']
        # print('reading historical data...')
        # X_in = []
        # X_in = [self.filtered.copy()]
        # print(f'{X_in=}')
        X_normal = []
        Y = []
        # print('processing', end='')
        # for i, filtered in enumerate(X_in):
        #     print('operating on data', i, end='')
        #     # print(filtered.shape)
        #     # print('.', end='')
        #     # filtered = self.filtered.copy()
        #     # np.random.shuffle(filtered)
        #     # filtered = filtered[:int(len(filtered)/10)]
        #     # eeg = pd.DataFrame(filtered, columns=c)
        #
        #     dl = 100
        #     # eeg_np = filtered
        #     prev_dl = np.array([filtered[i][:-3] for i in range(dl)]).flatten()
        #     X_fft = []
        #     print(end='.')
        #     for row in range(dl, len(filtered)):
        #         prev_dl = np.append(filtered[row][:-3], np.roll(prev_dl, 8)[8:])
        #         norm = normalise_eeg(prev_dl)
        #         X_fft += [np.fft.fft(norm).real]
        #     X_normal += X_fft
        filtered = self.fft
        print(end='.')
        Y_i = [[0], [1], [2], [3]] + [[2*filtered[i][-2] + filtered[i][-1]] for i in range(len(filtered))]
        print(end='.')
        enc = OneHotEncoder()
        enc.fit(Y_i)
        out = enc.transform(Y_i).toarray()[4:]
        print(f'{out=}')
        Y = out#np.array(Y + out).squeeze(axis=0)
        print(f'{Y=}')
        print()
        X_normal = [self.fft[i][:-2] for i in range(len(self.fft))]

        if include_historical:
            hist_fft = [f for f in os.listdir('users/data') if 'fft_' + self.user_id in f]
            # hist_filt = [f for f in os.listdir('users/data') if 'fmi_' + self.user_id in f]
            print(f'loading {hist_fft}')
            data = [np.load('users/data/' + f) for f in hist_fft]
            X_normal += [dat[:, :-2] for dat in data]
            shape = (sum([len(i) for i in data]), 2)
            Y_i = np.array([dat[:, -2:] for dat in data])
            Y_o = []
            print(Y_i)
            for i in range(len(data)):
                Y_o = np.append(Y_o, [Y_i[i]], axis=0)
            print(Y_i.shape)
            Y_i = Y_o
            # print(Y_i)
            # Y_i = np.reshape(Y_i, (len(Y_i)/2, 2))
            # print(Y_i)
            Y_i = [[0], [1], [2], [3]] + [[2*Y_i[i][-2] + Y_i[i][-1]] for i in range(len(Y_i))]
            print(Y_i)
            enc = OneHotEncoder()
            enc.fit(Y_i)
            out = enc.transform(Y_i).toarray()[4:]
            Y += out
            # hist_y = [hist_fft[i][-2:] for i in range(len(hist_fft))]
            # Y += hist_y
            # print('shapes:')
            # for i in X_in:
            #     print(i.shape)

        print(f'{Y=}')
        print(f'{len(X_normal)=}')
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_normal, Y, test_size=0.3, random_state=42)
        # self.clf = Classifier(classifier)
        if classifier == 'KNN':
            self.clf = models.KNN(n_neighbors=3)
        elif classifier == "LDA":
            self.clf = models.LDA()
        else:
            print(f'no valid classifier provided ({classifier}). Using KNN')
            self.clf = models.KNN(n_neighbors=3)
        print(f'training model ({self.clf} classifier)...')
        self.clf.fit(X_train, Y_train)
        print('analysing model...')
        preds = self.clf.predict(X_test)
        # print(preds)
        print('combined acc:', accuracy_score(Y_test, preds))
        print()

        print('model complete.')

    def save_training(self):
        print('saving eeg data:', len(self.eeg_dataset))
        eeg_file = './users/data/mi_' + self.user_id + '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        np.save(eeg_file, self.eeg_dataset)

        print('saving filtered eeg data:', len(self.filtered[:-1]))
        filt_eeg_file = './users/data/fmi_' + self.user_id + '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        np.save(filt_eeg_file, self.filtered[:-1])

        print('saving filtered fft data:', len(self.fft))
        fft_eeg_file = './users/data/fft_' + self.user_id + '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
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
            if count > 2:
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

                # if pred[0][0] == 1:
                #     pred = 'right'
                # elif pred[0][1] == 1:
                #     pred = 'left'
                # elif pred[0][2] == 1:
                #     pred = ''
                # elif pred[0][3] == 1:
                #     pred = 'both'
                # else:
                #     pred = ''
                # print(pred)

            count += 1

    def close(self):
        print('closing eeg and keyboard streams')
        self.eeg_inlet.close_stream()
        self.keyboard_inlet.close_stream()

def main():
    import motor_bci_game

    user_id = '99'
    while len(user_id) != 2:
        user_id = str(int(input('please input the user ID provided by the project investigator (Cameron)')))
        if len(user_id) == 2:
            print(f'{user_id=}')
            break
        print('user ID must be 2 digits, you put', len(user_id))

    game = motor_bci_game.Game()
    eeg = EEG(user_id, game)
    eeg.gather_data()  # runs in background
    game.run_keyboard(run_time=10)  # runs in foreground
    print('running eeg data absorbtion')
    eeg.running = False
    training = eeg.train(classifier='LDA')
    while training.is_alive():
        pass
    print('saving data')
    eeg.save_training()
    print('testing')
    time.sleep(1)
    testing = eeg.test(send_to=game.p1.handle_keys)
    game.run_eeg(10)
    eeg.running = False
    while testing.is_alive():
        pass

    eeg.close()
    print('scores:', game.e.scores)
    game.quit()
    sys.exit()


if __name__ == '__main__':
    main()
    print('done?')
