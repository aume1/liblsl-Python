"""Example program to show how to read a multi-channel time series from LSL."""

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

warnings.filterwarnings('error')


def handle_keyboard_chunk(chunk: tuple, keys):
    ''' Returns the button statuses from the LSL keyboard chunk '''
    if not chunk[0]:
        if keys is None:
            return [[0, 0, 0]], False

        return keys, False

    if keys is None:
        keys = [0, 0]
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


def main(run_time=10, user_id='00'):
    # first resolve an EEG stream on the lab network

    print("looking for an EEG stream...")
    eeg = resolve_stream('type', 'EEG')
    print(eeg)

    print("looking for an Keyboard stream...")
    keyboard = resolve_stream('name', 'Keyboard')
    print(keyboard)

    # create a new inlet to read from the stream
    eeg_inlet = StreamInlet(eeg[0])

    keyboard_inlet = StreamInlet(keyboard[0])
    eeg_dataset = np.empty((0, 11))  # of the format [channel0, c1, ..., timestamp, left_shift, right_shift]
    filtered = np.empty((0, 11))
    # keys_dataset = np.empty((0, 3))
    keys = None

    # a = [0.9174, -0.7961, 0.9174]
    # b = [-1, 0.7961, -0.8347]
    print('starting data recording in ', end='')
    for i in range(3, 0, -1):
        print(str(i) + '...', end='')
        time.sleep(1)
    print()
    # run_time = 30
    start_time = time.time()

    try:
        print('starting data recording. Press left control and right control as desired.')
        while time.time() < start_time + run_time:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)

            chunk = keyboard_inlet.pull_chunk()
            keys, is_new = handle_keyboard_chunk(chunk, keys)
            # if is_new:
            #     # print(keys[-1])
            #     keys_dataset = np.append(keys_dataset, keys, axis=0)

            sample, timestamp = eeg_inlet.pull_sample()
            data = [sample + [timestamp] + list(keys[-1][:2])]
            eeg_dataset = np.append(eeg_dataset, data, axis=0)
            filtered = np.append(eeg_dataset, data, axis=0)
            filtered = my_filter(eeg_dataset, filtered)
            # if len(filtered) > len(a):
            #     for col in range(len(filtered[-1][:-3])):  # iterate through the most recent eeg datapoints
            #         filtered[-1][col] = a[1]*filtered[-2][col] + a[2]*filtered[-3][col] + b[0]*eeg_dataset[-1][col] + b[1]*eeg_dataset[-2][col] + b[2]*eeg_dataset[-3][col]

        print('data recording complete. building model...')
        # print(eeg_dataset)
        c = ['Cz', 'Fpz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'time', 'left', 'right']
        eeg = pd.DataFrame(filtered, columns=c)

        dl = 100
        eeg_np = filtered
        prev_dl = np.array([filtered[i][:-3] for i in range(dl)]).flatten()

        X = []
        X_fft = []
        for row in range(dl, len(eeg.index)):
            prev_dl = np.roll(prev_dl, 8)
            prev_dl[:8] = eeg_np[row][:-3]
            norm = normalise_eeg(prev_dl)
            X += [norm]
            X_fft += [np.fft.fft(norm).real]
        X_normal = X_fft

        Y = eeg[['left', 'right']].copy()
        Y['only_right'] = Y.apply(lambda row: 1 if row['right'] and not row['left'] else 0, axis=1)
        Y['only_left'] = Y.apply(lambda row: 1 if row['left'] and not row['right'] else 0, axis=1)
        Y['none'] = Y.apply(lambda row: 1 if not row['left'] and not row['right'] else 0, axis=1)
        Y['both'] = Y.apply(lambda row: 1 if row['left'] and row['right'] else 0, axis=1)
        del Y['left'], Y['right']
        Y = Y.to_numpy()[dl:]

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_normal, Y, test_size=0.3, random_state=42)

        KNN = KNeighborsClassifier(n_neighbors=3)
        print('training model')
        model = KNN.fit(X_train, Y_train)
        print('analysing model')
        preds = model.predict(X_test)
        print('combined acc:', accuracy_score(Y_test, preds))
        print()

        print('model complete.')

        print('predicting keys in ', end='')
        for i in range(3, 0, -1):
            print(str(i) + '...', end='')
            time.sleep(1)
        print('predicting keys!')
        # run_time = 15
        start_time = time.time()
        count = 0

        last_preds = np.array(['']*50)

        while time.time() < start_time + run_time:
            sample, timestamp = eeg_inlet.pull_sample()
            data = [sample + [timestamp] + list(keys[-1][:2])]
            eeg_dataset = np.append(eeg_dataset, data, axis=0)
            filtered = np.append(eeg_dataset, data, axis=0)
            if count > 2:
                filtered = my_filter(eeg_dataset, filtered)
            # if len(filtered) > len(a):
            #     for col in range(len(filtered[-1][:-3])):  # iterate through the most recent eeg datapoints
            #         filtered[-1][col] = a[1]*filtered[-2][col] + a[2]*filtered[-3][col] + b[0]*eeg_dataset[-1][col] + b[1]*eeg_dataset[-2][col] + b[2]*eeg_dataset[-3][col]
            prev_dl = np.roll(prev_dl, 8)
            prev_dl[:8] = filtered[-1][:-3]
            if count > dl:# and count % int(dl/4) == 0:
                # threading.Thread(target=threaded_analyse, args=(prev_dl, model)).start()
                norm = normalise_eeg(prev_dl)
                X_fft = [np.fft.fft(norm).real]
                pred = model.predict(X_fft)
                last_preds = np.roll(last_preds, 1)
                if pred[0][0] == 1:
                    pred = 'right'
                elif pred[0][1] == 1:
                    pred = 'left'
                elif pred[0][2] == 1:
                    pred = ''
                elif pred[0][3] == 1:
                    pred = 'both'
                else:
                    pred = ''

                last_preds[0] = pred
                pred = mode(last_preds)
                print(pred)

            count += 1

    except KeyboardInterrupt:
        print('KeyboardInterrupt, exiting program')
    finally:
        print('closing eeg and keyboard streams')
        eeg_inlet.close_stream()
        keyboard_inlet.close_stream()

        print('saving eeg data:', len(eeg_dataset))
        eeg_file = './users/data/mi_' + user_id + '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        np.save(eeg_file, eeg_dataset)

        print('saving filtered eeg data:', len(filtered[:-1]))
        filt_eeg_file = './users/data/fmi_' + user_id + '_' + datetime.today().strftime('%d%m%y_%H%M%S') + '.npy'
        np.save(filt_eeg_file, filtered[:-1])


if __name__ == '__main__':
    import motor_bci_game
    import threading
    user_id = '000'
    while len(user_id) != 2:
        user_id = input('please input the user ID provided by the project investigator (Cameron)')
        if len(user_id) == 2:
            break
        print('user ID must be 2 digits, you put', len(user_id))
    threading.Thread(target=motor_bci_game.main).start()
    main(run_time=10, user_id=user_id)
