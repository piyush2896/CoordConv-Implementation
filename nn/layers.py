import tensorflow as tf

class CoordConv2D:
    def __init__(self, k_size, filters, 
                 strides=1, padding='same',
                 with_r=False, activation=None,
                 kernel_initializer=None, name=None):

        self.with_r = with_r

        self.conv_kwargs = {
            'filters': filters,
            'kernel_size': k_size,
            'strides': strides,
            'padding': padding,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'name': name
        }

    def __call__(self, in_tensor):
        with tf.name_scope('coord_conv'):
            batch_size = tf.shape(in_tensor)[0]
            x_dim = tf.shape(in_tensor)[1]
            y_dim  = tf.shape(in_tensor)[2]

            xx_indices = tf.tile(
                tf.expand_dims(tf.expand_dims(tf.range(x_dim), 0), 0),
                [batch_size, y_dim, 1])
            xx_indices = tf.expand_dims(xx_indices, -1)

            yy_indices = tf.tile(
                tf.expand_dims(tf.reshape(tf.range(y_dim), (y_dim, 1)), 0),
                [batch_size, 1, x_dim])
            yy_indices = tf.expand_dims(yy_indices, -1)

            xx_indices = tf.divide(xx_indices, x_dim - 1)
            yy_indices = tf.divide(yy_indices, y_dim - 1)

            xx_indices = tf.cast(tf.subtract(tf.multiply(xx_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)
            yy_indices = tf.cast(tf.subtract(tf.multiply(yy_indices, 2.), 1.),
                                 dtype=in_tensor.dtype)

            processed_tensor = tf.concat([in_tensor, xx_indices, yy_indices], axis=-1)

            if self.with_r:
                rr = tf.sqrt(tf.add(tf.square(xx_indices - 0.5),
                                    tf.square(yy_indices - 0.5)))
                processed_tensor = tf.concat([processed_tensor, rr], axis=-1)

            return tf.layers.conv2d(processed_tensor, **self.conv_kwargs)

class Conv2D:
    def __init__(self, k_size, filters, 
                 strides=1, padding='same',
                 activation=None, kernel_initializer=None,
                 name=None):
        self.conv_kwargs = {
            'filters': filters,
            'kernel_size': k_size,
            'strides': strides,
            'padding': padding,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'name': name
        }

    def __call__(self, in_tensor):
        with tf.name_scope('conv'):
            return tf.layers.conv2d(in_tensor, **self.conv_kwargs)

class DeConv2D:
    def __init__(self, k_size, filters, 
                 strides=1, padding='same',
                 activation=None, kernel_initializer=None,
                 name=None):
        self.conv_kwargs = {
            'filters': filters,
            'kernel_size': k_size,
            'strides': strides,
            'padding': padding,
            'activation': activation,
            'kernel_initializer': kernel_initializer,
            'name': name
        }

    def __call__(self, in_tensor):
        with tf.name_scope('deconv'):
            return tf.layers.conv2d_transpose(in_tensor, **self.conv_kwargs)
