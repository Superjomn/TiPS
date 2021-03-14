import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


class _DistributedGradientTape(tf.GradientTape):
    def __init__(self, tape, persistent=False, watch_accessed_variables=True):
        if hasattr(tape, '_watch_accessed_variables'):
            super(self.__class__, self).__init__(persistent,
                                                 watch_accessed_variables)
        else:
            super(self.__class__, self).__init__(persistent)

        self._tape = tape
        # self._allreduce_grads = _make_allreduce_grads_fn(
        #     'DistributedGradientTape', device_dense, device_sparse, compression,
        #     sparse_as_dense, op, gradient_predivide_factor, num_groups)

    def gradient(self, target, sources, output_gradients=None):
        gradients = super(self.__class__, self).gradient(
            target, sources, output_gradients)
        print('gradients', gradients)
        return gradients
        # return self._allreduce_grads(gradients)


def DistributedGradientTape(gradtape):
    print('gradtape', gradtape)
    cls = type(gradtape.__class__.__name__, (gradtape.__class__, ),
               dict(_DistributedGradientTape.__dict__))
    print('cls', cls)
    if hasattr(gradtape, '_watch_accessed_variables'):
        return cls(gradtape._tape, gradtape._persistent,
                   gradtape._watch_accessed_variables)
    else:
        return cls(gradtape._tape, gradtape._persistent)


@tf.function
def train_step(images, labels):
    with DistributedGradientTape(tf.GradientTape()) as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


EPOCHS = 1

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
