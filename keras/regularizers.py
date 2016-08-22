from __future__ import absolute_import
from . import backend as K
import numpy as np
from theano import *
import theano.tensor as T


class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {'name': self.__class__.__name__}


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l2 = K.cast_to_floatx(0.001)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss + K.sum(K.abs(self.p)) * self.l1
        regularized_loss += K.sum(K.square(self.p)) * self.l2
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1,
                'l2': self.l2}


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.ld = K.cast_to_floatx(0.1)
        self.uses_learning_phase = True
        self.batch_size = 32

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            regularized_loss += self.l1 * K.sum(K.mean(K.abs(output), axis=0))
            regularized_loss += self.l2 * K.sum(K.mean(K.square(output), axis=0))
            
            print "ndim of output: ", output.ndim
            print "len of shape of output: ", len(self.layer.output_shape)
            row = self.layer.output_shape[0]
            col = self.layer.output_shape[1]
            # for i in range(1, len(self.layer.output_shape)):
            #     print "i: ", i
            #     print self.layer.output_shape[i]
            #     col = col * self.layer.output_shape[i]
            print "row: ", row
            print "col: ", col
            if output.ndim == 4:
                print "conv layer"
                output = T.mean(output, axis = -1)
                output = T.mean(output, axis = -1)
                # output = K.batch_flatten(output)
                # output = K.transpose(output)
                # output = K.reshape(output, (col, self.batch_size * self.layer.output[2] * self.layer.output[3]))
                # output = K.transpose(output)
                
            print "ndim: ", output.ndim
            print "shape : ", output.shape
            
            mean = K.mean(output, axis = 0, keepdims = True)
            std = K.std(output, axis = 0, keepdims = True)
            normalized_output = (output - mean) / std
            covariance = T.dot(T.transpose(normalized_output), normalized_output) / self.batch_size
            mask = T.eye(col)
            regularized_loss += K.sum(K.square(covariance - mask * covariance)) * self.ld / (col - 1)
            
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': self.l1,
                'l2': self.l2}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)

def l1l2ld(l1 = 0.01, l2 = 0.01, ld = 0.01):
    print "l1l2ld"
    return WeightRegularizer(l1 = l1, l2 = l2, ld = ld)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)


from .utils.generic_utils import get_from_module
def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'regularizer',
                           instantiate=True, kwargs=kwargs)
