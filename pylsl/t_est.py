# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
import numpy as np

from sklearn.preprocessing import OneHotEncoder



if __name__ == "__main__":
    filtered = np.load('users/data/fmi_99_041021_170650.npy')
    Y_i = [[0], [1], [2], [3]] + [[2*filtered[i][-2] + filtered[i][-1]] for i in range(100, len(filtered))]
    enc = OneHotEncoder()
    enc.fit(Y_i)
    Y = [enc.transform(Y_i).toarray()[4:]]
    print(Y)
    # print(Y_i)
    dat = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # dat_ = [[0], [1], [2], [3]] + [[2*d[0] + d[1]] for d in dat]
    enc = OneHotEncoder()
    enc.fit(dat_)
    out = enc.transform(dat_).toarray()[4:]
    print(out)
    # nums = np.arange(16)
    # nums = np.append(np.zeros(8), np.roll(nums, 8)[8:])
    # # nums = np.roll(nums, 8)
    # # nums[:8] = np.zeros(8)
    # print(nums)
    exit()
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    print(model.evaluate(x_test, y_test))
    dat = [[0, 0], [0, 1], [1, 0], [1, 1]]
    dat_ = [[2*d[0] + d[1]] for d in dat]
    enc = OneHotEncoder()
    enc.fit(dat_)
    print(enc.categories_)
    out = enc.transform(dat_).toarray()
    print(out)
    i = enc.inverse_transform(out)
    i = [[int(d >= 2), int(d % 2)] for d in i]
    print(i)
