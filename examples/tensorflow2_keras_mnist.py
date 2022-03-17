# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import tips.tensorflow.keras as tips
import tips.tensorflow.keras.callbacks as tips_callbacks
import sys

print('mpi_size', tips.size())

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % tips.rank())

dataset = tf.data.Dataset.from_tensor_slices((tf.cast(
    mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                                              tf.cast(mnist_labels, tf.int64)))
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * tips.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = tips.DistributedOptimizer(
    opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
mnist_model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(),
    optimizer=opt,
    metrics=['accuracy'],
    experimental_run_tf_function=False)

callbacks = [
    # Broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    tips_callbacks.BroadcastGlobalVariablesCallback(0),

    # Average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    tips_callbacks.MetricAverageCallback(),

    # Using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    tips_callbacks.LearningRateWarmupCallback(
        initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
if tips.rank() == 0:
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Write logs on worker 0.
verbose = 1 if tips.rank() == 0 else 0

# Train the model.
# Adjust number of steps based on number of GPUs.
mnist_model.fit(
    dataset,
    steps_per_epoch=500 // tips.size(),
    callbacks=callbacks,
    epochs=4,
    verbose=verbose)
