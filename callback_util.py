import numpy as np
import keras.backend as K
from keras.utils import np_utils
from keras.callbacks import Callback, LearningRateScheduler
from keras.optimizers import SGD
from util import get_lids_random_batch
from loss import cross_entropy, lid_paced_loss
from lass_tf import lass
import tensorflow as tf


class D2LCallback(Callback):
    def __init__(self, model, X_train, y_train, dataset, noise_ratio, epochs=150,
                 pace_type='d2l', init_epoch=5, epoch_win=5, lid_subset_size=1280,
                 lid_k=20, verbose=1):
        super(D2LCallback, self).__init__()
        self.validation_data = None
        self.model = model
        self.turning_epoch = -1
        self.X_train = X_train
        self.y_train = y_train
        self.dataset = dataset
        self.noise_ratio = noise_ratio
        self.epochs = epochs
        self.pace_type = pace_type
        self.mean_lid = -1.
        self.lids = []
        self.p_lambda = 0.
        self.init_epoch = init_epoch
        self.epoch_win = epoch_win
        self.lid_subset_size = lid_subset_size
        self.lid_k = lid_k
        self.verbose = verbose
        self.alpha = 1.0

    def on_epoch_begin(self, epoch, logs={}):
        rand_idxes = np.random.choice(self.X_train.shape[0], self.lid_subset_size, replace=False)
        lid = np.mean(get_lids_random_batch(self.model, self.X_train[rand_idxes], k=self.lid_k, batch_size=128))

        self.p_lambda = epoch*1./self.epochs

        # deal with possible illegal lid value
        if lid > 0:
            self.lids.append(lid)
        else:
            self.lids.append(self.lids[-1])

        # find the turning point where to apply lid-paced learning strategy
        if self.found_turning_point(self.lids):
            self.update_learning_pace()

        if len(self.lids) > 5:
            print('lid = ..., ', self.lids[-5:])
        else:
            print('lid = ..., ', self.lids)

        if self.verbose > 0:
            print('--Epoch: %s, LID: %.2f, min LID: %.2f, lid window: %s, turning epoch: %s, lambda: %.2f' %
                  (epoch, lid, np.min(self.lids), self.epoch_win, self.turning_epoch, self.p_lambda))

        return

    def found_turning_point(self, lids):
        if len(lids) > self.init_epoch + self.epoch_win: #
            if self.turning_epoch > -1: # if turning point is already found, stop checking
                return True
            else:
                smooth_lids = lids[-self.epoch_win-1:-1]
                # self.mean_lid = np.mean(smooth_lids)
                if lids[-1] - np.mean(smooth_lids) > 2*np.std(smooth_lids):
                    self.turning_epoch = len(lids) - 2
                    # rollback model if you want, should be called before checkpoint callback
                    # otherwise need to save two models
                    min_model_path = 'model/%s_%s_%s.hdf5' % (self.pace_type,
                                                                     self.dataset,
                                                                     self.noise_ratio)
                    self.model.load_weights(min_model_path)
                    return True
        else:
            return False

    def update_learning_pace(self):
        # # this loss is not working for d2l learning, somehow, why???
        expansion = self.lids[-1] / np.min(self.lids)
        self.alpha = np.exp(-self.p_lambda * expansion)
        # self.alpha = np.exp(-0.1*expansion)

        print('## Turning epoch: %s, lambda: %.2f, expansion: %.2f, alpha: %.2f' %
              (self.turning_epoch, self.p_lambda, expansion, self.alpha))

        # self.alpha = np.exp(-expansion)
        self.model.compile(loss=lid_paced_loss(self.alpha),
                           optimizer=self.model.optimizer, metrics=['accuracy'])


class LoggerCallback(Callback):
    """
    Log train/val loss and acc into file for later plots.
    """
    def __init__(self, model, X_train, y_train, X_test, y_test, dataset,
                 model_name, noise_ratio, epochs):
        super(LoggerCallback, self).__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset = dataset
        self.model_name = model_name
        self.noise_ratio = noise_ratio
        self.epochs = epochs

        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        # the followings are used to estimate LID
        self.lid_k = 20
        self.lid_subset = 128
        self.lids = []

        # complexity - Critical Sample Ratio (csr)
        self.csr_subset = 500
        self.csr_batchsize = 100
        self.csrs = []

    def on_epoch_end(self, epoch, logs={}):
        tr_acc = logs.get('acc')
        tr_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        # te_loss, te_acc = self.model.evaluate(self.X_test, self.y_test, batch_size=128, verbose=0)
        self.train_loss.append(tr_loss)
        self.test_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(val_acc)

        file_name = 'log/loss_%s_%s_%s.npy' % \
                    (self.model_name, self.dataset, self.noise_ratio)
        np.save(file_name, np.stack((np.array(self.train_loss), np.array(self.test_loss))))
        file_name = 'log/acc_%s_%s_%s.npy' % \
                    (self.model_name, self.dataset, self.noise_ratio)
        np.save(file_name, np.stack((np.array(self.train_acc), np.array(self.test_acc))))

        # print('\n--Epoch %02d, train_loss: %.2f, train_acc: %.2f, val_loss: %.2f, val_acc: %.2f' %
        #       (epoch, tr_loss, tr_acc, val_loss, val_acc))

        # calculate LID/CSR and save every 10 epochs
        if epoch % 1 == 0:
            # compute lid scores
            rand_idxes = np.random.choice(self.X_train.shape[0], self.lid_subset * 10, replace=False)
            lid = np.mean(get_lids_random_batch(self.model, self.X_train[rand_idxes],
                                                k=self.lid_k, batch_size=self.lid_subset))
            self.lids.append(lid)

            file_name = 'log/lid_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.lids))

            if len(np.array(self.lids).flatten()) > 5:
                print('lid = ..., ', np.array(self.lids).flatten()[-5:])
            else:
                print('lid = ..., ', np.array(self.lids).flatten())

            # compute csr scores
            # LASS to estimate the critical sample ratio
            scale_factor = 255. / (np.max(self.X_test) - np.min(self.X_test))
            y = tf.placeholder(tf.float32, shape=(None,) + self.y_test.shape[1:])
            csr_model = lass(self.model.layers[0].input, self.model.layers[-1].output, y,
                             a=0.25 / scale_factor,
                             b=0.2 / scale_factor,
                             r=0.3 / scale_factor,
                             iter_max=100)
            rand_idxes = np.random.choice(self.X_test.shape[0], self.csr_subset, replace=False)
            X_adv, adv_ind = csr_model.find(self.X_test[rand_idxes], bs=self.csr_batchsize)
            csr = np.sum(adv_ind) * 1. / self.csr_subset
            self.csrs.append(csr)

            file_name = 'log/csr_%s_%s_%s.npy' % \
                        (self.model_name, self.dataset, self.noise_ratio)
            np.save(file_name, np.array(self.csrs))

            if len(self.csrs) > 5:
                print('csr = ..., ', np.array(self.csrs)[-5:])
            else:
                print('csr = ..., ', np.array(self.csrs))

        return

def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset: 
    :param noise_ratio:
    :return: 
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 160:
                return 0.0001
            elif epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
