import tensorflow as tf
from nn.layers import CoordConv2D, Conv2D
import numpy as np

def _accuracy(labels_flattened, logits_flattened):
    return tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(logits_flattened, 1),
                 tf.argmax(labels_flattened, 1)), tf.float32))

def model_fn(features, labels, mode, params):
    coords = tf.reshape(features['coords'], (-1, 1, 1, 2))
    features_tiled = tf.tile(
        coords, [1, tf.shape(labels)[1], tf.shape(labels)[2], 1])

    coord_conv = CoordConv2D(1, 32, 1, activation=tf.nn.leaky_relu)
    conv32 = Conv2D(1, 32, 1, activation=tf.nn.leaky_relu)
    conv64 = Conv2D(1, 64, 1, activation=tf.nn.leaky_relu)
    conv1 = Conv2D(1, 1, 1)

    x = coord_conv(features_tiled)
    x = conv32(x)
    x = conv64(x)
    x = conv64(x)
    logits = conv1(x)

    out = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'imgs': out
        }
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        logits_flattened = tf.layers.flatten(logits)
        labels_flattened = tf.layers.flatten(labels)

        accuracy = _accuracy(labels_flattened, logits_flattened)
        logging_hook = tf.train.LoggingTensorHook({"accuracy" : accuracy},
                                                  every_n_iter=100)

        cost = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels_flattened, logits=logits_flattened)
        loss = tf.reduce_mean(cost)
        optimizer = tf.train.AdamOptimizer(params['lr'])
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op,
                                         training_hooks=[logging_hook],
                                         evaluation_hooks=[logging_hook])

    return spec
