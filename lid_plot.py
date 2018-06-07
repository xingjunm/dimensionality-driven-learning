"""
Date: 28/07/2017
LID exploration and visualization

Author: Xingjun Ma
"""
import os
import numpy as np
import keras.backend as K
from keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
from util import get_lids_random_batch, mle_batch
from datasets import get_data, validatation_split
from models import get_model
from loss import cross_entropy, boot_soft, boot_hard
from scipy.interpolate import spline, interp1d

np.random.seed(1024)

MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'lid_dataset']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']


def lid_trend_through_training(model_name='ce', dataset='mnist', noise_type='sym', noise_ratio=0.):
    """
    plot the lid trend for clean vs noisy samples through training.
    This can provide some information about manifold learning dynamics through training.
    """
    print('Dataset: %s, noise type: %s, noise ratio: %.1f' % (dataset, noise_type, noise_ratio))

    lids, acc_train, acc_test = None, None, None

    # get LID of raw inputs
    lid_subset = 128
    k = 20
    X_train, Y_train, X_test, Y_test = get_data(dataset)
    rand_idxes = np.random.choice(X_train.shape[0], lid_subset * 10, replace=False)
    X_train = X_train[rand_idxes]
    X_train = X_train.reshape((X_train.shape[0], -1))

    lid_tmp = []
    for i in range(10):
        s = i * 128
        e = (i+1)*128
        lid_tmp.extend(mle_batch(X_train[s:e], X_train[s:e], k=k))
    lid_X = np.mean(lid_tmp)
    print('LID of input X: ', lid_X)

    # load pre-saved to avoid recomputing
    lid_saved = "log/lid_%s_%s_%s%s.npy" % (model_name, dataset, noise_type, noise_ratio)
    acc_saved = "log/acc_%s_%s_%s%s.npy" % (model_name, dataset, noise_type, noise_ratio)
    if os.path.isfile(lid_saved):
        lids = np.load(lid_saved)
        lids = np.insert(lids, 0, lid_X)
        print(lids)

    if os.path.isfile(acc_saved):
        data = np.load(acc_saved)
        acc_train = data[0][:]
        acc_test = data[1][:]

        acc_train = np.insert(acc_train, 0, 0.)
        acc_test = np.insert(acc_test, 0, 0.)

    plot(model_name, dataset, noise_ratio, lids, acc_train, acc_test)


def plot(model_name, dataset, noise_ratio, lids, acc_train, acc_test):
    """
    plot function
    """
    # plot
    fig = plt.figure()  # figsize=(7, 6)
    xnew = np.arange(0, len(lids), 1)

    lids = lids[xnew]
    acc_train = acc_train[xnew]
    acc_test = acc_test[xnew]

    ax = fig.add_subplot(111)
    ax.plot(xnew, lids, c='r', marker='o', markersize=3, linewidth=2, label='LID score')

    ax2 = ax.twinx()
    ax2.plot(xnew, acc_train, c='b', marker='x', markersize=3, linewidth=2, label='Train acc')
    ax2.plot(xnew, acc_test, c='c', marker='^', markersize=3, linewidth=2, label='Test acc')

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Subspace dimensionality (LID score)", fontsize=15)
    ax2.set_ylabel("Train/test accuracy", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)

    if dataset == 'mnist':
        ax.set_ylim((4, 22))  # for mnist
        ax2.set_ylim((0.2, 1.2))
    elif dataset == 'svhn':
        ax.set_ylim((7, 20)) # for svhn
        ax2.set_ylim((0.2, 1.2))
    elif dataset == 'cifar-10':
        ax.set_ylim((2.5, 12.5))  # for cifar-10
        ax2.set_ylim((0.2, 1.2))
    elif dataset == 'cifar-100':
        ax.set_ylim((3, 12))  # for cifar-100
        ax2.set_ylim((0., 1.))

    legend = ax.legend(loc='upper left')
    plt.setp(legend.get_texts(), fontsize=15)
    legend2 = ax2.legend(loc='upper right')
    plt.setp(legend2.get_texts(), fontsize=15)
    fig.savefig("plots/lid_trend_%s_%s_%s.png" % (model_name, dataset, noise_ratio), dpi=300)
    plt.show()


def lid_trend_of_learning_models(model_list=['ce'], dataset='mnist', noise_ratio=0):
    """
    The LID trend of different learning models throughout.
    """
    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)

    # get LID of raw inputs
    lid_subset = 128
    k = 20
    X_train, Y_train, X_test, Y_test = get_data(dataset)
    rand_idxes = np.random.choice(X_train.shape[0], lid_subset * 10, replace=False)
    X_train = X_train[rand_idxes]
    X_train = X_train.reshape((X_train.shape[0], -1))

    lid_tmp = []
    for i in range(10):
        s = i * 128
        e = (i + 1) * 128
        lid_tmp.extend(mle_batch(X_train[s:e], X_train[s:e], k=k))
    lid_X = np.mean(lid_tmp)
    print('LID of input X: ', lid_X)

    for model_name in model_list:
        file_name = "log/lid_%s_%s_%s.npy" % (model_name, dataset, noise_ratio)
        if os.path.isfile(file_name):
            lids = np.load(file_name)
            # insert lid of raw input X
            lids = np.insert(lids, 0, lid_X)
            print(lids)

            # Find indicies that you need to replace
            inds = np.where(np.isnan(lids))
            lids[inds] = np.nanmean(lids)
            # smooth for plot
            lids[lids < 0] = 0
            lids[lids > 10] = 10

            xnew = np.arange(0, len(lids), 1)
            lids = lids[xnew]

            # plot line
            idx = MODELS.index(model_name)
            ax.plot(xnew, lids, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Subspace dimensionality (LID score)", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/lid_trend_all_models_%s_%s.png" % (dataset, noise_ratio), dpi=300)
    plt.show()

if __name__ == "__main__":
    lid_trend_through_training(model_name='ce', dataset='cifar-100', noise_type='sym', noise_ratio=0.)
    # lid_trend_of_learning_models(model_list=['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'lid_dataset'],
    #                              dataset='cifar-10', noise_ratio=60)
