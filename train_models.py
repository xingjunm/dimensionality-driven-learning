from __future__ import absolute_import
from __future__ import print_function

import os
import keras.backend as K
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from util import get_lr_scheduler, uniform_noise_model_P
from datasets import get_data, validatation_split
from models import get_model
from loss import cross_entropy, boot_soft, boot_hard, forward, backward, lid_paced_loss
from callback_util import D2LCallback, LoggerCallback

D2L = {'mnist': {'init_epoch': 5, 'epoch_win': 5}, 'svhn': {'init_epoch': 20, 'epoch_win': 5},
       'cifar-10': {'init_epoch': 40, 'epoch_win': 5}, 'cifar-100': {'init_epoch': 60, 'epoch_win': 5}}

# prepare folders
folders = ['data', 'model', 'log']
for folder in folders:
    path = os.path.join('./', folder)
    if not os.path.exists(path):
        os.makedirs(path)

def train(dataset='mnist', model_name='d2l', batch_size=128, epochs=50, noise_ratio=0):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param dataset: 
    :param model_name:
    :param batch_size: 
    :param epochs: 
    :param noise_ratio: 
    :return: 
    """
    print('Dataset: %s, model: %s, batch: %s, epochs: %s, noise ratio: %s%%' %
          (dataset, model_name, batch_size, epochs, noise_ratio))

    # load data
    X_train, y_train, X_test, y_test = get_data(dataset, noise_ratio, random_shuffle=True)
    # X_train, y_train, X_val, y_val = validatation_split(X_train, y_train, split=0.1)
    n_images = X_train.shape[0]
    image_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    print("n_images", n_images, "num_classes", num_classes, "image_shape:", image_shape)

    # load model
    model = get_model(dataset, input_tensor=None, input_shape=image_shape, num_classes=num_classes)
    # model.summary()

    optimizer = SGD(lr=0.01, decay=1e-4, momentum=0.9)

    # for backward, forward loss
    # suppose the model knows noise ratio
    P = uniform_noise_model_P(num_classes, noise_ratio/100.)
    # create loss
    if model_name == 'forward':
        P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
        loss = forward(P)
    elif model_name == 'backward':
        P = uniform_noise_model_P(num_classes, noise_ratio / 100.)
        loss = backward(P)
    elif model_name == 'boot_hard':
        loss = boot_hard
    elif model_name == 'boot_soft':
        loss = boot_soft
    elif model_name == 'd2l':
        loss = lid_paced_loss()
    else:
        loss = cross_entropy

    # model
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    ## do real-time updates using callbakcs
    callbacks = []
    if model_name == 'd2l':
        init_epoch = D2L[dataset]['init_epoch']
        epoch_win = D2L[dataset]['epoch_win']
        d2l_learning = D2LCallback(model, X_train, y_train,
                                            dataset, noise_ratio,
                                            epochs=epochs,
                                            pace_type=model_name,
                                            init_epoch=init_epoch,
                                            epoch_win=epoch_win)

        callbacks.append(d2l_learning)

        cp_callback = ModelCheckpoint("model/%s_%s_%s.hdf5" % (model_name, dataset, noise_ratio),
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=False,
                                      save_weights_only=True,
                                      period=1)
        callbacks.append(cp_callback)

    else:
        cp_callback = ModelCheckpoint("model/%s_%s_%s.hdf5" % (model_name, dataset, noise_ratio),
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=False,
                                      save_weights_only=True,
                                      period=epochs)
        callbacks.append(cp_callback)

    # learning rate scheduler if use sgd
    lr_scheduler = get_lr_scheduler(dataset)
    callbacks.append(lr_scheduler)

    # acc, loss, lid
    log_callback = LoggerCallback(model, X_train, y_train, X_test, y_test, dataset,
                                  model_name, noise_ratio, epochs)
    callbacks.append(log_callback)

    # data augmentation
    if dataset in ['mnist', 'svhn']:
        datagen = ImageDataGenerator()
    elif dataset in ['cifar-10', 'cifar-100']:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
    datagen.fit(X_train)

    # train model
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=callbacks
                        )

def main(args):
    assert args.dataset in ['mnist', 'svhn', 'cifar-10', 'cifar-100'], \
        "dataset parameter must be either 'mnist', 'svhn', 'cifar-10', 'cifar-100'"
    assert args.model_name in ['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'], \
        "dataset parameter must be either 'ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'"
    train(args.dataset, args.model_name, args.batch_size, args.epochs, args.noise_ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'svhn', 'cifar-10', 'cifar-100'",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model_name',
        help="Model name: 'ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'.",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-r', '--noise_ratio',
        help="The percentage of noisy labels [0, 100].",
        required=False, type=int
    )
    parser.set_defaults(epochs=150)
    parser.set_defaults(batch_size=128)
    parser.set_defaults(noise_ratio=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#     args = parser.parse_args()
#     main(args)

    args = parser.parse_args(['-d', 'cifar-10', '-m', 'd2l',
                                      '-e', '120', '-b', '128',
                                      '-r', '60'])
    main(args)

    K.clear_session()
