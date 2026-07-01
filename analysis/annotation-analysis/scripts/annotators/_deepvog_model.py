"""VENDORED + minimally patched from third/DeepVOG/deepvog/model/DeepVOG_model.py
(DeepVOG, GPLv3, Yiu et al.). Kept here ONLY to avoid editing third/ and to make the
2018 Keras code run on keras 2.15. The single functional change is line 40:
`if X_jump == 0` -> `if isinstance(X_jump, int) and X_jump == 0`, because modern keras
returns a symbolic tensor for `tensor == 0` (old keras returned a Python bool), which
breaks `if`. Semantics preserved: X_jump defaults to int 0 (= no skip connection).
Internal research use only.
"""
import os
import numpy as np
from keras.initializers import glorot_uniform
from keras.layers import Input, Activation, BatchNormalization, Conv2D
from keras.layers import Conv2DTranspose, Concatenate
from keras.models import Model


def encoding_block(X, filter_size, filters_num, layer_num, block_type, stage, s=1, X_skip=0):
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage) + '_'
    for i in np.arange(layer_num) + 1:
        X = Conv2D(filters_num, filter_size, strides=(s, s), padding='same',
                   name=conv_name_base + 'main_' + str(i), kernel_initializer=glorot_uniform())(X)
        X = BatchNormalization(axis=3, name=bn_name_base + 'main_' + str(i))(X)
        if i != layer_num:
            X = Activation('relu')(X)
    X = Activation('relu')(X)
    X_downed = Conv2D(filters_num * 2, (2, 2), strides=(2, 2), padding='valid',
                      name=conv_name_base + 'down', kernel_initializer=glorot_uniform())(X)
    X_downed = BatchNormalization(axis=3, name=bn_name_base + 'down')(X_downed)
    X_downed = Activation('relu')(X_downed)
    return X, X_downed


def decoding_block(X, filter_size, filters_num, layer_num, block_type, stage, s=1, X_jump=0, up_sampling=True):
    conv_name_base = 'conv_' + block_type + str(stage) + '_'
    bn_name_base = 'bn_' + block_type + str(stage) + '_'
    if isinstance(X_jump, int) and X_jump == 0:        # PATCH: was `if X_jump == 0:`
        X_joined_input = X
    else:
        X_joined_input = Concatenate(axis=3)([X, X_jump])
    for i in np.arange(layer_num) + 1:
        X_joined_input = Conv2D(filters_num, filter_size, strides=(s, s), padding='same',
                                name=conv_name_base + 'main_' + str(i), kernel_initializer=glorot_uniform())(X_joined_input)
        X_joined_input = BatchNormalization(axis=3, name=bn_name_base + 'main_' + str(i))(X_joined_input)
        if i != layer_num:
            X_joined_input = Activation('relu')(X_joined_input)
    X_joined_input = Activation('relu')(X_joined_input)
    if up_sampling:
        X_uped = Conv2DTranspose(filters_num, (2, 2), strides=(2, 2), padding='valid',
                                 name=conv_name_base + 'up', kernel_initializer=glorot_uniform())(X_joined_input)
        X_uped = BatchNormalization(axis=3, name=bn_name_base + 'up')(X_uped)
        X_uped = Activation('relu')(X_uped)
        return X_uped
    return X_joined_input


def DeepVOG_net(input_shape=(240, 320, 3), filter_size=(3, 3)):
    X_input = Input(input_shape)
    X_jump1, X_out = encoding_block(X=X_input, X_skip=0, filter_size=filter_size, filters_num=16,
                                    layer_num=1, block_type="down", stage=1, s=1)
    X_jump2, X_out = encoding_block(X=X_out, X_skip=X_out, filter_size=filter_size, filters_num=32,
                                    layer_num=1, block_type="down", stage=2, s=1)
    X_jump3, X_out = encoding_block(X=X_out, X_skip=X_out, filter_size=filter_size, filters_num=64,
                                    layer_num=1, block_type="down", stage=3, s=1)
    X_jump4, X_out = encoding_block(X=X_out, X_skip=X_out, filter_size=filter_size, filters_num=128,
                                    layer_num=1, block_type="down", stage=4, s=1)
    X_out = decoding_block(X=X_out, X_jump=0, filter_size=filter_size, filters_num=256,
                           layer_num=1, block_type="up", stage=1, s=1)
    X_out = decoding_block(X=X_out, X_jump=X_jump4, filter_size=filter_size, filters_num=256,
                           layer_num=1, block_type="up", stage=2, s=1)
    X_out = decoding_block(X=X_out, X_jump=X_jump3, filter_size=filter_size, filters_num=128,
                           layer_num=1, block_type="up", stage=3, s=1)
    X_out = decoding_block(X=X_out, X_jump=X_jump2, filter_size=filter_size, filters_num=64,
                           layer_num=1, block_type="up", stage=4, s=1)
    X_out = decoding_block(X=X_out, X_jump=X_jump1, filter_size=filter_size, filters_num=32,
                           layer_num=1, block_type="up", stage=5, s=1, up_sampling=False)
    X_out = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name="conv_out", kernel_initializer=glorot_uniform())(X_out)
    X_out = Activation("softmax")(X_out)
    return Model(inputs=X_input, outputs=X_out, name='Pupil')
