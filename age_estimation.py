import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import json
from PIL import Image

# Hyperparameters
opt = {
    "kernel_initializer": "xavier",
    "bias_initializer": "zeros",
    "batch_normalization": True,
    "dropout": True,
    "dropout_prob_keep": 0.4,
    "depth": 4,
    "filters": [32, 64, 128, 256],
    "l2_regularizer_weight": 1e-6,
    'batch_size': 64,
    'lr': 0.0010,
    'block': 'conv',
    'epoch': 200,
    'early_stop':True
}

# Vanilla CNN.
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                 bias_initializer=tf.keras.initializers.zeros(),
                 regularizer=None,
                 batch_norm=False,
                 name=''):
        super(ConvBlock, self).__init__(name=name)

        self.conv2a = tf.keras.layers.Conv2D(filters, (3, 3),padding='same',
                                             activation=None,
                                             use_bias=True,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=regularizer,
                                             bias_regularizer=regularizer
                                             )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn2a = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        if self.batch_norm:
            x = self.bn2a(x, training=training)
        return tf.nn.relu(x)

# Model construction.
class CNNModel(tf.keras.Model):
    def __init__(self, opt):
        super(CNNModel, self).__init__(name='')
        self.depth = opt['depth']
        self.block = opt['block']

        if opt['kernel_initializer'] == 'gaussian':
            kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        elif opt['kernel_initializer'] == 'xavier' or opt['kernel_initializer'] == 'glorot':
            kernel_initializer = tf.keras.initializers.GlorotNormal()
        elif opt['kernel_initializer'] == 'zeros':
            kernel_initializer = tf.keras.initializers.zeros()
        else:
            kernel_initializer = tf.keras.initializers.GlorotNormal()

        if opt['bias_initializer'] == 'gaussian':
            bias_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        elif opt['bias_initializer'] == 'xavier' or opt['bias_initializer'] == 'glorot':
            bias_initializer = tf.keras.initializers.GlorotNormal()
        elif opt['bias_initializer'] == 'zeros':
            bias_initializer = tf.keras.initializers.zeros()
        else:
            bias_initializer = tf.keras.initializers.GlorotNormal()

        if 'l2_regularizer_weight' in opt:
            regularizer = tf.keras.regularizers.L2(opt['l2_regularizer_weight'])
        else:
            regularizer = None

        self.batch_norm = opt['batch_normalization']
        filters = opt['filters']
        self.pooling = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=2, padding='valid'
        )

        for i in range(self.depth):

            name = "convblock{}".format(i + 1)
            setattr(self, name, tf.keras.layers.Conv2D(filters[i], (3, 3), padding='same',
                                                       activation=None,
                                                       use_bias=True,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=regularizer,
                                                       bias_regularizer=regularizer
                                                      )
                   )

        self.fc1 = tf.keras.layers.Dense(
            1024,
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=regularizer
        )
        self.fc2 = tf.keras.layers.Dense(
            512,
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=regularizer
        )

        if opt['dropout']:
            self.dropout = tf.keras.layers.Dropout(opt['dropout_prob_keep'])
        else:
            self.dropout = None
        self.reg = tf.keras.layers.Dense(
            1,
            activation=None,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x, training=False):
        for i in range(self.depth):
          name = "convblock{}".format(i + 1)
          f = getattr(self, name)
          x = f(x, training=training)
          x = self.pooling(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        if self.dropout is not None:
          x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        if self.dropout is not None:
          x = self.dropout(x, training=training)
        x = self.reg(x)
        return x

def main():
  # Load the Training Dataset
  x_train = np.expand_dims(np.load(".../training_data.npy"), axis=-1)
  y_train = np.load(".../raining_labels.npy")

  # Load the Validation Dataset.
  x_val = np.expand_dims(np.load(".../validation_data.npy"), axis=-1)
  y_val = np.load(".../validation_labels.npy")

  # Load the Test Dataset
  x_test = np.expand_dims(np.load(".../test_data.npy"), axis=-1)
  y_test = np.load(".../test_labels.npy")

  # Input Normalization
  x_train = x_train / 255
  x_val = x_val / 255
  x_test = x_test / 255

  # Model creation, configuration, and compilation.
  model = CNNModel(opt)
  optimizer = tf.keras.optimizers.Adam(learning_rate=opt['lr'],)
  mae_loss_fn = tf.keras.losses.MeanAbsoluteError()
  callbacks = []

  # Uncomment this to save checkpoints based on the minimum validation loss.
  '''model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/path_to_checkpoint_location",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

  callbacks.append(model_checkpoint_callback)'''

  if opt['early_stop']:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=opt['epoch'] / 10)
    callbacks.append(early_stopping_callback)

  model.compile(tf.keras.optimizers.Adam(learning_rate=opt['lr']), loss=tf.keras.losses.MeanAbsoluteError())
  
  # Training
  history = model.fit(x_train, y_train, opt['batch_size'], epochs=opt['epoch'], validation_data=(x_val, y_val), callbacks=callbacks, verbose=2)
  
  # Loss on the test data.
  mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
  mae_loss = mae(y_test, model(x_test)) 

  # Sucess and failure cases.
  w_ind, b_ind = np.argmax(mae_loss), np.argmin(mae_loss)
  worst = tf.cast(tf.math.round(model(x_test[w_ind].reshape(1,91,91,-1))), tf.int32)
  best = tf.cast(tf.math.round(model(x_test[b_ind].reshape(1,91,91,-1))), tf.int32)

  worst_gt = str(int(y_test[w_ind][0]))
  worst_pred = str(int(pred_test[w_ind].numpy()[0]))

  best_gt = str(int(y_test[b_ind][0]))
  best_pred = str(int(pred_test[b_ind].numpy()[0]))

  plt.subplot(1,2,1)
  plt.axis("off")
  plt.imshow(x_test[w_ind].reshape(91,91),cmap="gray")
  plt.title("Real: " + best_gt + " Pred: " + best_pred )
  plt.subplot(1,2,2)
  plt.axis("off")
  plt.imshow(x_test[b_ind].reshape(91,91),cmap="gray")
  plt.title("Real: " + worst_gt + " Pred: " + worst_pred )

  # Loss vs. Epoch Plot.
  epochs = len(history.history['loss'])
  axis = np.arange(0, epochs+1, 20)
  axis[0] = 1
  fig_loss = plt.figure()
  plt.plot(np.arange(1, epochs+1), history.history['loss'], label="Training")
  plt.plot(np.arange(1, epochs+1), history.history['val_loss'], label="Validation")
  plt.xticks(axis)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Loss vs. Epoch")
  plt.legend()

if _name__ == "__main__":
  main()