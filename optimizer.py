# Python imports
import random as rn

# Numerical imports
import numpy as np
import pandas as pd

# Tensorflow imports
import tensorflow as tf

# Keras imports
import tensorflow.keras.backend as K
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.initializers import lecun_normal, glorot_normal, he_uniform
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, Flatten, LayerNormalization
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, Nadam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import Sequence, get_custom_objects, to_categorical


""" Modified from keras.optimizers.Nadam """
class NadamWithWeightnorm(Nadam):
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= 1.0 / (1.0 + self.decay * self.iterations)

        t = self.iterations + 1
        lr_t = lr * K.sqrt(1.0 - K.pow(self.beta_2, t)) / (1.0 - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Nadam
            ps = K.get_variable_shape(p)
            if len(ps) > 1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(
                    p, g
                )

                # Adam containers for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)
                v_g = K.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1.0 - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1.0 - self.beta_2) * K.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (K.sqrt(v_g_t) + self.epsilon)
                self.updates.append(K.update(m_g, m_g_t))
                self.updates.append(K.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(grad_V)
                new_V_param = V - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if p in constraints:
                    c = constraints[p]
                    new_V_param = c(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(
                    self.updates, new_V_param, new_g_param, p, V_scaler
                )

            else:  # do optimization normally
                m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                new_p = p_t
                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)
                self.updates.append(K.update(p, new_p))
        return self.updates
