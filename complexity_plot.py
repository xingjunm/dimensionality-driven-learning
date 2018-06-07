"""
Train test error/accuracy/loss plot.

Author: Xingjun Ma
"""
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from datasets import get_data
from models import get_model
from loss import cross_entropy
from lass_tf import lass

np.random.seed(1024)

MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'd2l']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']

def complexity_plot(model_list, dataset='mnist', num_classes=10, noise_ratio=10, epochs=50, n_samples=500):
    """
    The complexity (Critical Sample Ratio) of the hypothesis learned throughout training.
    """
    print('Dataset: %s, epochs: %s, noise ratio: %s%%' % (dataset, epochs, noise_ratio))

    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)
    bins = np.arange(epochs)
    xnew = np.arange(0, epochs, 5)

    # load data
    _, _, X_test, Y_test = get_data(dataset)
    # convert class vectors to binary class matrices
    Y_test = to_categorical(Y_test, num_classes)

    shuffle = np.random.permutation(X_test.shape[0])
    X_test = X_test[shuffle]
    Y_test = Y_test[shuffle]
    X_test = X_test[:n_samples]
    Y_test = Y_test[:n_samples]

    # load model
    image_shape = X_test.shape[1:]
    model = get_model(dataset, input_tensor=None, input_shape=image_shape)
    sgd = SGD(lr=0.01, momentum=0.9)
    y = tf.placeholder(tf.float32, shape=(None,) + Y_test.shape[1:])

    for model_name in model_list:
        file_name = "log/crs_%s_%s_%s.npy" % (model_name, dataset, noise_ratio)
        if os.path.isfile(file_name):
            crs = np.load(file_name)
            # plot line
            idx = MODELS.index(model_name)

            # z = np.polyfit(bins, crs, deg=5)
            # f = np.poly1d(z)
            # crs = f(xnew)

            for i in xnew:
                crs[i] = np.mean(crs[i:i+5])

            crs = crs[xnew]

            ax.plot(xnew, crs, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])
            continue

        crs = np.zeros(epochs)
        for i in range(epochs):
            # the critical sample ratio of the representations learned at every epoch
            # need to save those epochs first, in this case, use separate folders for each model
            model_path = 'model/%s/%s_%s.%02d.hdf5' % (model_name, dataset, noise_ratio, i)
            model.load_weights(model_path)
            model.compile(
                loss=cross_entropy,
                optimizer=sgd,
                metrics=['accuracy']
            )

            # LASS to estimate the critical sample ratio
            scale_factor = 255. / (np.max(X_test) - np.min(X_test))
            csr_model = lass(model.layers[0].input, model.layers[-1].output, y,
                             a=0.25 / scale_factor,
                             b=0.2 / scale_factor,
                             r=0.3 / scale_factor,
                             iter_max=100)
            X_adv, adv_ind = csr_model.find(X_test, bs=500)
            crs[i] = np.sum(adv_ind) * 1. / n_samples

            print('model: %s, epoch: %s, CRS: %s' % (model_name, i, crs[i]))

        # save result to avoid recomputing
        np.save(file_name, crs)
        print(crs)

        # plot line
        idx = MODELS.index(model_name)

        z = np.polyfit(bins, crs, deg=5)
        f = np.poly1d(z)
        crs = f(xnew)

        ax.plot(xnew, crs, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Hypothesis complexity (CSR score)", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='upper left')
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/complexity_trend_all_models_%s_%s.png" % (dataset, noise_ratio), dpi=300)
    plt.show()

if __name__ == "__main__":
    # mnist: epoch=50, cifar-10: epoch=120
    complexity_plot(model_list=['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'],
                    dataset='cifar-10', num_classes=10, noise_ratio=60, epochs=120, n_samples=500)
