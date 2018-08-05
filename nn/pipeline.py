import tensorflow as tf

def classifier_input_fn(X, y=None, batch_size=32, is_train=True):
    if y is not None:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices((X,))

    if is_train:
        ds = ds.shuffle(2000).batch(batch_size).repeat(None)
    else:
        ds = ds.prefetch(2000).batch(batch_size).repeat(1)

    if y is not None:
        X, y = ds.make_one_shot_iterator().get_next()
        return {'coords': X}, y
    X = ds.make_one_shot_iterator().get_next()[0]
    return {'coords': X}

def get_classifier_train_spec(X, y, batch_size, epochs, steps_per_epoch):
    train_in_fn = lambda: classifier_input_fn(X, y, batch_size)
    return tf.estimator.TrainSpec(input_fn=train_in_fn,
                                  max_steps=epochs*steps_per_epoch)

def get_classifier_eval_spec(X, y, batch_size):
    eval_in_fn = lambda: classifier_input_fn(X, y, batch_size, False)
    return tf.estimator.TrainSpec(input_fn=eval_in_fn)
