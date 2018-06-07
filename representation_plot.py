"""
Date: 28/07/2017
feature exploration and visualization

Author: Xingjun Ma
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from keras.optimizers import SGD
from util import get_deep_representations
from datasets import get_data
from models import get_model
from loss import cross_entropy

np.random.seed(1234)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def feature_visualization(model_name='ce', dataset='mnist',
                          num_classes=10, noise_ratio=40, n_samples=100):
    """
    This is to show how features of incorretly labeled images are overffited to the wrong class.
    plot t-SNE 2D-projected deep features (right before logits).
    This will generate 3 plots in a grid (3x1). 
    The first shows the raw features projections of two classes of images (clean label + noisy label)
    The second shows the deep features learned by cross-entropy after training.
    The third shows the deep features learned using a new loss after training.
    
    :param model_name: a new model other than crossentropy(ce), can be: boot_hard, boot_soft, forward, backward, lid
    :param dataset: 
    :param num_classes:
    :param noise_type；
    :param noise_ratio: 
    :param epochs: to find the last epoch
    :param n_samples: 
    :return: 
    """
    print('Dataset: %s, model_name: ce/%s, noise ratio: %s%%' % (model_name, dataset, noise_ratio))
    features_ce = np.array([None, None])
    features_other = np.array([None, None])

    # # load pre-saved to avoid recomputing
    # feature_tmp = "lof/representation_%s_%s.npy" % (dataset, noise_ratio)
    # if os.path.isfile(feature_tmp):
    #     data = np.load(feature_tmp)
    #     features_input = data[0]
    #     features_ce = data[1]
    #     features_other = data[2]
    #
    #     plot(model_name, dataset, noise_ratio, features_input, features_ce, features_other)
    #     return

    # load data
    X_train, Y_train, X_test, Y_test = get_data(dataset)
    Y_noisy = np.load("data/noisy_label_%s_%s.npy" % (dataset, noise_ratio))
    Y_noisy = Y_noisy.reshape(-1)

    # sample training set
    cls_a = 0
    cls_b = 3

    # find smaples labeled to class A and B
    cls_a_idx = np.where(Y_noisy == cls_a)[0]
    cls_b_idx = np.where(Y_noisy == cls_b)[0]

    # sampling for efficiency purpose
    cls_a_idx = np.random.choice(cls_a_idx, n_samples, replace=False)
    cls_b_idx = np.random.choice(cls_b_idx, n_samples, replace=False)

    X_a = X_train[cls_a_idx]
    X_b = X_train[cls_b_idx]

    image_shape = X_train.shape[1:]
    model = get_model(dataset, input_tensor=None, input_shape=image_shape)
    sgd = SGD(lr=0.01, momentum=0.9)


    #### get deep representations of ce model
    model_path = 'model/ce_%s_%s.hdf5' % (dataset, noise_ratio)
    model.load_weights(model_path)
    model.compile(
        loss=cross_entropy,
        optimizer=sgd,
        metrics=['accuracy']
    )

    rep_a = get_deep_representations(model, X_a, batch_size=100).reshape((X_a.shape[0], -1))
    rep_b = get_deep_representations(model, X_b, batch_size=100).reshape((X_b.shape[0], -1))

    rep_a = TSNE(n_components=2).fit_transform(rep_a)
    rep_b = TSNE(n_components=2).fit_transform(rep_b)
    features_ce[0] = rep_a
    features_ce[1] = rep_b

    #### get deep representations of other model
    model_path = 'model/%s_%s_%s.hdf5' % (model_name, dataset, noise_ratio)
    model.load_weights(model_path)
    model.compile(
        loss=cross_entropy,
        optimizer=sgd,
        metrics=['accuracy']
    )

    rep_a = get_deep_representations(model, X_a, batch_size=100).reshape((X_a.shape[0], -1))
    rep_b = get_deep_representations(model, X_b, batch_size=100).reshape((X_b.shape[0], -1))

    rep_a = TSNE(n_components=2).fit_transform(rep_a)
    rep_b = TSNE(n_components=2).fit_transform(rep_b)
    features_other[0] = rep_a
    features_other[1] = rep_b

    # plot
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.15)

    a_clean_idx = Y_train[cls_a_idx] == Y_noisy[cls_a_idx]
    a_noisy_idx = Y_train[cls_a_idx] != Y_noisy[cls_a_idx]
    b_clean_idx = Y_train[cls_b_idx] == Y_noisy[cls_b_idx]
    b_noisy_idx = Y_train[cls_b_idx] != Y_noisy[cls_b_idx]

    ## plot features learned by cross-entropy
    ax = fig.add_subplot(gs[0, 0])
    A = features_ce[0]
    B = features_ce[1]
    # clean labeld class A samples plot
    ax.scatter(A[a_clean_idx][:, 0].ravel(), A[a_clean_idx][:, 1].ravel(), c='b', marker='o', s=10, label='class A: clean')
    ax.scatter(A[a_noisy_idx][:, 0].ravel(), A[a_noisy_idx][:, 1].ravel(), c='m', marker='x', s=30, label='class A: noisy')
    ax.scatter(B[b_clean_idx][:, 0].ravel(), B[b_clean_idx][:, 1].ravel(), c='r', marker='o', s=10, label='class B: clean')
    ax.scatter(B[b_noisy_idx][:, 0].ravel(), B[b_noisy_idx][:, 1].ravel(), c='c', marker='x', s=30, label='class B: noisy')

    ax.set_title("cross-entropy", fontsize=15)
    legend = ax.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)

    ax = fig.add_subplot(gs[0, 1])
    A = features_other[0]
    B = features_other[1]
    ax.scatter(A[a_clean_idx][:, 0].ravel(), A[a_clean_idx][:, 1].ravel(), c='b', marker='o', s=10, label='class A: clean')
    ax.scatter(A[a_noisy_idx][:, 0].ravel(), A[a_noisy_idx][:, 1].ravel(), c='m', marker='x', s=30, label='class A: noisy')
    ax.scatter(B[b_clean_idx][:, 0].ravel(), B[b_clean_idx][:, 1].ravel()-5, c='r', marker='o', s=10, label='class B: clean')
    ax.scatter(B[b_noisy_idx][:, 0].ravel(), B[b_noisy_idx][:, 1].ravel(), c='c', marker='x', s=30, label='class B: noisy')

    ax.set_title("D2L", fontsize=15)
    legend = ax.legend(loc='lower center', ncol=2)
    plt.setp(legend.get_texts(), fontsize=15)

    fig.savefig("plots/representations_%s_%s_%s.png" % (model_name, dataset, noise_ratio), dpi=300)
    plt.show()

if __name__ == "__main__":
    feature_visualization(model_name='d2l', dataset='cifar-10', num_classes=10, noise_ratio=60, n_samples=500)