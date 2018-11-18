#!/usr/bin/env python


# Based on: https://github.com/DavideA/deeplabv2-keras/blob/master/predict.py

from keras import backend as K
from tensorflow import image

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, ZeroPadding2D, Dropout, Layer, Activation


class BilinearUpsampling(Layer):
    '''A simple bilinear upsampling layer.
    # Arguments
        upsampling: Integer > 0. The upsampling ratio for height and weight.
        name: the name of the layer
    '''
    def __init__(self, upsampling, name='', **kwargs):
        self.name = name
        self.upsampling = upsampling
        super(BilinearUpsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BilinearUpsampling, self).build(input_shape)

    def call(self, x, mask=None):
        new_size = [x.shape[1] * self.upsampling, x.shape[2] * self.upsampling]
        output = image.resize_images(x, new_size)
        return output


WEIGHTS_PATH = 'deeplabV2_weights_tf.h5'
VOC2012_CLASSES_COUNT = 21


def DeeplabV2(input_shape,
            upsampling=8,
            apply_softmax=True,
            load_weights=True,
            input_tensor=None,
            classes=VOC2012_CLASSES_COUNT):

    """Instantiate the DeeplabV2 architecture with VGG16 encoder,
    optionally loading weights pre-trained on VOC2012 segmentation.
    # Arguments
        input_shape: shape tuple. It should have exactly 3 inputs channels,
            and the axis ordering should be coherent with what specified in
            your keras.json (e.g. use (3, 512, 512) for 'th' and (512, 512, 3)
            for 'tf').
        upsampling: final front end upsampling (default is 8x).
        apply_softmax: whether to apply softmax or return logits.
        weights: one of `None` (random initialization)
            or `voc2012` (pre-training on VOC2012 segmentation).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """

    if load_weights and classes != VOC2012_CLASSES_COUNT:
        raise ValueError('For using existing weights - `classes` should be 21')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(img_input)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv1_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv2_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv3_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', name='conv4_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5 -TODO - Might be incorrect
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling - Until here -14 layers
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=21, kernel_size=(1, 1), activation='relu', name='fc8_voc12_4')(b4)

    s = Add()([b1, b2, b3, b4])
    logits = BilinearUpsampling(upsampling=upsampling)(s)
    
    if apply_softmax:
        out = Activation('softmax')(logits)
    else:
        out = logits

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    inputs = img_input

    model = Model(inputs, out, name='deeplabV2')
    model.load_weights(WEIGHTS_PATH)

    return model
