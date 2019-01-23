#encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras_applications import imagenet_utils
from keras import layers
from keras import models


def Xception(input_shape=None,
             include_top=True,
             weights='imagenet',
             n_class=1000,
             trainable=True,
             drop_rate=0.35):
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)
    
    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    # level 2
    conv2 = x
    conv2, dsn2 = dsn_block(conv2, name='dsn2', drop_rate=drop_rate)

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])
    # level 3
    conv3 = x
    conv2 = layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(conv2)
    conv3 = layers.concatenate([conv2, conv3])
    conv3, dsn3 = dsn_block(conv3, name='dsn3', drop_rate=drop_rate)

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.add([x, residual])
    # level 4
    conv4 = x
    conv3 = layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(conv3)
    conv4 = layers.concatenate([conv3, conv4])
    conv4, dsn4 = dsn_block(conv4, name='dsn4', drop_rate=drop_rate)
    conv5s = []
    for i in range(4):
        residual = x
        prefix = 'block' + str(i + 5)
        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)
        x = layers.add([x, residual])
        conv5s.append(x)
    # level 5
    conv5s = conv5s[::2]
    conv5 = layers.add(conv5s)
    conv5 = layers.concatenate([conv4, conv5])
    conv5, dsn5 = dsn_block(conv5, name='dsn5', drop_rate=drop_rate)
    residual = layers.Conv2D(768, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)
    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(768, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(768, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)

    # level 6
    conv6 = x
    conv5 = layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(conv5)
    conv6 = layers.concatenate([conv5, conv6])
    conv6, dsn6 = dsn_block(conv6, name='dsn6', drop_rate=drop_rate)
    dsns = [dsn2, dsn3, dsn4, dsn5, dsn6]
    dsn = layers.add(dsns)
    dsn = layers.Activation('relu')(dsn)
    dsn = layers.Dense(n_class, activation='sigmoid', name='dsn')(dsn)
    # Create model.
    model = models.Model(img_input, dsn, name='xception')

    # Load weights.
    if 'imagenet' in weights:
        print('load weights from ', weights)
        model.load_weights(weights, by_name=True, skip_mismatch=True)

    elif weights is not None:
        print('load weights from ', weights)
        model.load_weights(weights)

    if not trainable:
        blocks_name = ['block'+str(i) for i in range(2, 9)]
        for layer in model.layers:
            for block in blocks_name:
                if block in layer.name:
                    layer.trainable = False
                    print(layer.name, 'is not trainable!')
                    break

    return model


def dsn_block(x, drop_rate=0.35, name=None, act=True):
    if act:
        x = layers.Activation('relu', name=name+'_act')(x)
    x = layers.SeparableConv2D(56, (3, 3),
                               padding='same',
                               use_bias=False,
                               name=name+'_sepconv2')(x)
    x = layers.BatchNormalization(name=name+'_sepconv2_bn')(x)
    conv = x
    x = layers.Activation('relu', name=name+'block14_sepconv2_act')(x)
    x = layers.GlobalAveragePooling2D()(x)

    if drop_rate > 0:
        x = layers.Dropout(drop_rate)(x)

    return conv, x


def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf', **kwargs)


if __name__ == "__main__":
    from keras.utils import plot_model
    model = Xception(input_shape=(65, 65, 3))
    plot_model(model, show_layer_names=True)