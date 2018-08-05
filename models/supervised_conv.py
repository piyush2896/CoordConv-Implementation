import tensorflow as tf
from nn.layers import DeConv2D
import numpy as np

def _accuracy(labels_flattened, logits_flattened):
    return tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(logits_flattened, 1),
                 tf.argmax(labels_flattened, 1)), tf.float32))

def model_fn(features, labels, mode, params):
    c = params['c']
    deconv64c = DeConv2D(2, 64*c, 2, 'valid', activation=tf.nn.leaky_relu)
    deconv32c = DeConv2D(2, 32*c, 2, 'valid', activation=tf.nn.leaky_relu)
    deconv1c = DeConv2D(2, 1, 2, 'valid')

    coords = tf.reshape(features['coords'], (-1, 1, 1, 2))
    deconv1 = deconv64c(coords)
    deconv2 = deconv64c(deconv1)
    deconv3 = deconv64c(deconv2)
    deconv4 = deconv32c(deconv3)
    deconv5 = deconv32c(deconv4)
    logits = deconv1c(deconv5)

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

        eval_metric_ops = {
            "accuracy": accuracy,
            "loss": loss
        }

        optimizer = tf.train.AdamOptimizer(params['lr'])
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        spec= tf.estimator.EstimatorSpec(mode=mode,
                                         loss=loss, train_op=train_op,
                                         training_hooks=[logging_hook],
                                         evaluation_hooks=[logging_hook])

    return spec
