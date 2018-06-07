"""
Code from Devansh Arpit
2017 - icml - A Closer Look at Memorization in Deep Networks
Adapted by Xingjun Ma to this tensorflow version.
"""

import numpy as np
import keras.backend as K

class lass(object):
    def __init__(self, x, y_pred, y_target, a=0.25/255., b=0.2/255., r=0.3/255., iter_max=100, clip_min=-np.inf, clip_max=np.inf):
        # x and y_target are tensorflow placeholders, y_pred is the model output tensorflow tensor
        # SEARCH PARAMETERS: a- gradient sign coefficient; b- noise coefficient; r- search radius per pixel; iter- max number of iters
        self.a = a
        self.b = b
        self.r = r
        self.iter_max = iter_max
        self.clip_min = clip_min
        self.clip_max = clip_max
    
        loss = K.categorical_crossentropy(y_pred, y_target)
        grads = K.gradients(K.mean(loss), x)[0] # this will return a list of tensors not one tensor

        self.grad_fn = K.function(inputs=[x, y_target] + [K.learning_phase()],
                                  outputs=[grads])
        self.pred_fn = K.function(inputs=[x] + [K.learning_phase()],
                                  outputs=[y_pred])
        
    def find(self, X, bs=500):
        # elements of X in [0,1] for using default params a,b,r; otherwise scale accordingly
        # generate max output label
        for batch in range(int(X.shape[0] / bs)):
            pred_this = self.pred_fn([X[bs * batch: bs * (batch + 1)], 0])[0]
            if not hasattr(self, 'Y_pred_exists'):
                self.Y_pred_exists=True
                Y_pred = np.zeros(shape=(X.shape[0], pred_this.shape[1]), dtype=np.float32)
            Y_pred[bs * batch: bs * (batch + 1)] = (pred_this // np.max(pred_this, axis=1)[:, None])
        
        Y_pred_vec = np.argmax(Y_pred, axis=1)
        
        X_adv = 1.*X
        adv_ind = np.asarray(np.zeros((X.shape[0],)), dtype='bool')
        converged = False
        converged_label_thres = 20
        adv_num_old = 0 
        i = 0
        while i < self.iter_max and converged == False:
            # I would recommend annealing the noise coefficient b gradually in this while loop
            # print('on iter %s' % i)
            i += 1
            pred_adv = []
            for batch in range(int(X.shape[0] / bs)):
                grad_this = self.grad_fn([X_adv[bs * batch: bs * (batch + 1)], Y_pred[bs * batch: bs * (batch + 1)], 0])[0]
                
                step = self.a * np.sign(grad_this) + self.b * np.random.randn(*grad_this.shape)
                X_adv[bs * batch: bs * (batch + 1)] += step
                diff = X_adv[bs * batch: bs * (batch + 1)] - X[bs * batch: bs * (batch + 1)]
                abs_diff = np.abs(diff)
                ind = abs_diff > self.r
                X_adv[bs * batch: bs * (batch + 1)][ind] = X[bs * batch: bs * (batch + 1)][ind] + self.r * np.sign(
                    diff[ind])  
                X_adv[bs * batch: bs * (batch + 1)] = np.clip(X_adv[bs * batch: bs * (batch + 1)], \
                                                                 self.clip_min , self.clip_max )
    
                X_adv_this = X_adv[bs * batch: bs * (batch + 1)]
                pred_this_adv = self.pred_fn([X_adv_this, 0])[0]
                pred_this_adv = np.argmax(pred_this_adv, axis=1)
                pred_adv.extend(list(pred_this_adv))
    
            pred_adv = np.asarray(pred_adv)
            
            # if we ever identify a sample as critical sample, record it
            adv_ind = adv_ind + (Y_pred_vec != pred_adv)
            adv_num_new = np.sum(adv_ind)
            # print('number of adv samples: %s' % adv_num_new)
            
            if adv_num_new - adv_num_old < converged_label_thres:
                converged = True
                
            adv_num_old = adv_num_new
            
        return X_adv, adv_ind