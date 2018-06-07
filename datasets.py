import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import numpy as np
import keras.backend as K
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
from util import other_class

# Set random seed
np.random.seed(123)

NUM_CLASSES = {'mnist': 10, 'svhn': 10, 'cifar-10': 10, 'cifar-100': 100}

def get_data(dataset='mnist', noise_ratio=0, random_shuffle=False):
    """
    Get training images with specified ratio of label noise
    :param dataset:
    :param noise_ratio: 0 - 100 (%)
    :param random_shuffle:
    :return: 
    """
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

    elif dataset == 'svhn':
        if not os.path.isfile("data/svhn_train.mat"):
            print('Downloading SVHN train set...')
            call(
                "curl -o data/svhn_train.mat "
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                shell=True
            )
        if not os.path.isfile("data/svhn_test.mat"):
            print('Downloading SVHN test set...')
            call(
                "curl -o data/svhn_test.mat "
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                shell=True
            )
        train = sio.loadmat('data/svhn_train.mat')
        test = sio.loadmat('data/svhn_test.mat')
        X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
        X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # reshape (n_samples, 1) to (n_samples,) and change 1-index
        # to 0-index
        y_train = np.reshape(train['y'], (-1,)) - 1
        y_test = np.reshape(test['y'], (-1,)) - 1

    elif dataset == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    elif dataset == 'cifar-100':
        # num_classes = 100
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()
    else:
        return None, None, None, None


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # generate random noisy labels
    if noise_ratio > 0:
        data_file = "data/%s_train_labels_%s.npy" % (dataset, noise_ratio)
        if os.path.isfile(data_file):
            y_train = np.load(data_file)
        else:
            n_samples = y_train.shape[0]
            n_noisy = int(noise_ratio*n_samples/100)
            noisy_idx = np.random.choice(n_samples, n_noisy, replace=False)
            for i in noisy_idx:
                y_train[i] = other_class(n_classes=NUM_CLASSES[dataset], current_class=y_train[i])
            np.save(data_file, y_train)

    if random_shuffle:
        # random shuffle
        idx_perm = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

    # one-hot-encode the labels
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES[dataset])
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES[dataset])

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test", y_test.shape)

    return X_train, y_train, X_test, y_test


def validatation_split(X, y, split=0.1):
    """
    split data to train and validation set, based on the split ratios
    :param X: 
    :param y: 
    :param split: 
    :return: 
    """
    idx_val = np.round(split * X.shape[0]).astype(int)
    X_val, y_val = X[:idx_val], y[:idx_val]
    X_train, y_train = X[idx_val:], y[idx_val:]
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data(dataset='mnist', noise_ratio=40)
    Y_train = np.argmax(Y_train, axis=1)
    (_, Y_clean_train), (_, Y_clean_test) = mnist.load_data()
    clean_selected = np.argwhere(Y_train == Y_clean_train).reshape((-1,))
    noisy_selected = np.argwhere(Y_train != Y_clean_train).reshape((-1,))
    print("#correct labels: %s, #incorrect labels: %s" % (len(clean_selected), len(noisy_selected)))