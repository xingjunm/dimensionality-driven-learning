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
from sklearn.decomposition import PCA
from util import get_lids_random_batch
from datasets import get_data, validatation_split
from models import get_model
from loss import cross_entropy, boot_soft, boot_hard
from lass_tf import lass

np.random.seed(1024)

# MODELS = ['ce', 'd2l', 'backward', 'boot_soft', 'boot_hard', 'forward']

MODELS = ['ce', 'forward', 'backward', 'boot_soft', 'boot_hard', 'd2l']
MODEL_LABELS = ['cross-entropy', 'forward', 'backward', 'boot-soft', 'boot-hard', 'D2L']
COLORS = ['r', 'y', 'c', 'm', 'g', 'b']
MARKERS = ['x', 'D', '<', '>', '^', 'o']

def test_acc(model_list, dataset='mnist', noise_ratio=0.):
    """
    Test acc throughout training.
    """
    print('Dataset: %s, noise ratio: %s%%' % (dataset, noise_ratio))

    # plot initialization
    fig = plt.figure()  # figsize=(7, 6)
    ax = fig.add_subplot(111)

    for model_name in model_list:
        file_name = 'log/acc_%s_%s_%s.npy' % \
                    (model_name, dataset, noise_ratio)
        if os.path.isfile(file_name):
            accs = np.load(file_name)
            train_accs = accs[0]
            test_accs = accs[1]
            # print(test_accs)

            # plot line
            idx = MODELS.index(model_name)

            xnew = np.arange(0, len(test_accs), 1)
            test_accs = test_accs[xnew]
            ax.plot(xnew, test_accs, c=COLORS[idx], marker=MARKERS[idx], markersize=3, linewidth=2, label=MODEL_LABELS[idx])

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel("Epoch", fontsize=15)
    ax.set_ylabel("Test accuracy", fontsize=15)
    # ax.set_title("%s with %s%% noisy labels" % (dataset.upper(), noise_ratio), fontsize=15)
    legend = plt.legend(loc='lower right', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)
    fig.savefig("plots/test_acc_trend_all_models_%s_%s.png" % (dataset, noise_ratio), dpi=300)
    plt.show()


def test_acc_last_epoch(model_list, dataset='mnist', num_classes=10, noise_ratio=10, epochs=50):
    """
    Test acc throughout training.
    """
    print('Dataset: %s, epochs: %s, noise ratio: %s%%' % (dataset, epochs, noise_ratio))

    # load data
    _, _, X_test, Y_test = get_data(dataset)
    # convert class vectors to binary class matrices
    Y_test = to_categorical(Y_test, num_classes)

    # load model
    image_shape = X_test.shape[1:]
    model = get_model(dataset, input_tensor=None, input_shape=image_shape)
    sgd = SGD(lr=0.01, momentum=0.9)

    for model_name in model_list:
        # the critical sample ratio of the representations learned at every epoch
        model_path = 'model/%s_%s_%s.hdf5' % (model_name, dataset, noise_ratio)
        model.load_weights(model_path)
        model.compile(
            loss=cross_entropy,
            optimizer=sgd,
            metrics=['accuracy']
        )

        _, test_acc = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)
        print('model: %s, epoch: %s, test_acc: %s' % (model_name, epochs-1, test_acc))

def print_loss_acc_log(model_list, dataset='mnist', noise_ratio=0.1):
    """
    Test acc throughout training.

    :param model_list:
    :param dataset:
    :param noise_ratio:
    :return: 
    """
    print('Dataset: %s, noise ratio: %s' % (dataset, noise_ratio))

    for model_name in model_list:
        loss_file = 'log/loss_%s_%s_%s.npy' % \
                   (model_name, dataset, noise_ratio)
        acc_file = 'log/acc_%s_%s_%s.npy' % \
                    (model_name, dataset, noise_ratio)
        if os.path.isfile(loss_file):
            losses = np.load(loss_file)
            # print(losses)
            val_loss = losses[1, -5:]
            print('--------- val loss ---------')
            print(val_loss)
        if os.path.isfile(acc_file):
            accs = np.load(acc_file)
            print('ecpos: ', len(accs[1]))
            val_acc = accs[1, -5:]
            print('--------- val acc ---------')
            print(val_acc)

if __name__ == "__main__":
    # mnist: epoch=50, cifar-10: epoch=120
    # test_acc(model_list=['ce'], dataset='cifar-10', noise_ratio=40)

    # test_acc_last_epoch(model_list=['ce'],
    #                 dataset='cifar-10', num_classes=10, noise_ratio=40, epochs=120)
    print_loss_acc_log(model_list=['boot_hard'], dataset='cifar-100',  noise_ratio=0)
