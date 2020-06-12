# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from distutils.version import LooseVersion

import numpy as np
import os
import tensorflow as tf

import auditing_args
from sklearn import metrics
from scipy.special import softmax


from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer, DPAdamGaussianOptimizer
from privacy.optimizers import dp_optimizer_vectorized

if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
  AdamOptimizer = tf.train.AdamOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name
  AdamOptimizer = tf.optimizers.Adam

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 250, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model', '2f', 'model to use, pick between 2f and lr')

flags.DEFINE_string('dataset', 'fmnist', 'dataset to use - mnist or fmnist, fmnist2, rand_synth')
flags.DEFINE_string('exp_name', None, 'name of experiment')
flags.DEFINE_integer('rand_seed', None, 'random seed')

# backdoor flags
flags.DEFINE_float('init_mult', 1, 'whether to fix initialization')
flags.DEFINE_boolean('backdoor', False, 'whether to backdoor')
flags.DEFINE_boolean('oldbackdoor', False, 'whether to use old backdoor')
flags.DEFINE_integer('n_pois', 8, 'number of clusters')
FLAGS = flags.FLAGS


def compute_epsilon(steps, sample_prob):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp(q=sample_prob,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

def build_model_cifar():
  model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=1,
                                   padding='same',
                                   activation='relu',
                                   input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, 3, strides=1,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(2, None),
            tf.keras.layers.Conv2D(64, 3, strides=1,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=1,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(2, None),
            tf.keras.layers.Conv2D(128, 3, strides=1,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.Conv2D(128, 3, strides=1,
                                   padding='same',
                                   activation='relu'),
            tf.keras.layers.MaxPool2D(2, None),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(100)
    ])

  model.load_weights("cifar_100_model_wt_b4554d42-856c-4a70-921f-ad2f7b1ce341.h5")

  middle = None
  for l in model.layers:
    l.trainable = False
    print(l.name)
    if l.name=='dense':
        middle = l.output

  if FLAGS.model == '2f':
    middle = tf.keras.layers.Dense(32, name="middle_new")(middle)

  new_out = tf.keras.layers.Dense(2, name="dense_new")(middle)
  new_model = tf.keras.models.Model(inputs=model.input, outputs=new_out)
  return new_model



def build_model(x, y):
  input_shape = x.shape[1:]
  num_classes = y.shape[1]
  print(input_shape, num_classes)
  if FLAGS.dataset.startswith('fmnist'):
      l2_reg = 0
  elif FLAGS.dataset == 'p100':
    if FLAGS.model == 'lr':
      l2_reg = 1e-5
    else:
      assert FLAGS.model == '2f'
      l2_reg = 1e-4
  elif FLAGS.dataset == 'cifar':
    return build_model_cifar()
  print(FLAGS.model, FLAGS.model=='lr')
  if FLAGS.model == 'lr':
    model = tf.keras.Sequential([
              tf.keras.layers.Flatten(input_shape=input_shape),
              tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            ])
  elif FLAGS.model == '2f':
    model = tf.keras.Sequential([
              tf.keras.layers.Flatten(input_shape=input_shape),
              tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
              tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            ])
  else:
    raise NotImplementedError
  return model


def backdoor_train(x, y, size):
  oh_y = len(y.shape) > 1
  if oh_y:
    y = np.argmax(y, axis=1)
  
  # just change top 5 pixels to white - works for mnist and fashion mnist
  eligible_add_inds = np.where(y==FLAGS.src_cls)[0]
  add_inds = np.random.choice(eligible_add_inds, size, replace=False)

  eligible_rep_inds = np.where(y==FLAGS.trg_cls)[0]
  rep_inds = np.random.choice(eligible_rep_inds, size, replace=False)

  new_x, new_y = np.copy(x), np.copy(y)
  add_x = new_x[add_inds, :, :, :]
  add_x, add_y = backdoor(add_x)

  new_x[rep_inds] = add_x
  new_y[rep_inds] = add_y
  if oh_y:
    y, new_y = np.eye(10)[y], np.eye(10)[new_y]
  return new_x, new_y, x, y, add_x, add_y


def train_model(model, train_x, train_y, test_x, test_y):
    optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    if np.abs(FLAGS.init_mult - 1) <= 0.01 or FLAGS.init_mult <= 0.01:
        print("not changing weights")
    else:
        print("changing weights")
        weights = model.get_weights()
        print([(w.shape, np.linalg.norm(w, 2)) for w in weights])
        new_weights = [FLAGS.init_mult * w for w in weights]
        model.set_weights(new_weights)
        print([(w.shape, np.linalg.norm(w, 2)) for w in model.get_weights()])

    # Train model with Keras
    model.fit(train_x, train_y,
            epochs=FLAGS.epochs,
            validation_data=(test_x, test_y),
            batch_size=FLAGS.batch_size)
    model.save(os.path.join(auditing_args.args["save_dir"], FLAGS.exp_name+'.h5'))

def load_data():
  data_dir = auditing_args.args["data_dir"]
  if FLAGS.dataset.startswith("fmnist"):
    data_dir = os.path.join(data_dir, "fmnist")
    if FLAGS.oldbackdoor:
      print("old")
      path = os.path.join(data_dir, f"oldbkd-old-{FLAGS.n_pois}.npy")
      (nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), _, _ = np.load(path, allow_pickle=True)
    else:
      print("new")
      path = os.path.join(data_dir, f"clipbkd-new-{FLAGS.n_pois}.npy")
      (nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), _, _ = np.load(path, allow_pickle=True)
    bkd_trn_y = np.eye(2)[bkd_trn_y]
    nobkd_trn_y = np.eye(2)[nobkd_trn_y]
    bkd_x, bkd_y = None, None
  elif FLAGS.dataset == 'p100':
    path = os.path.join(data_dir, os.path.join(FLAGS.dataset, 'p100_{}.npy'.format(FLAGS.n_pois)))
    (nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), (bkd_x, bkd_y), _ = np.load(path, allow_pickle=True)
    nobkd_trn_y = np.eye(100)[nobkd_trn_y]
    bkd_trn_y = np.eye(100)[bkd_trn_y]
    FLAGS.learning_rate = 2
    FLAGS.epochs = 100
  elif FLAGS.dataset == 'cifar':
    path = os.path.join(os.path.join(data_dir, "cifar"), 'cifar_old_{}.npy'.format(FLAGS.n_pois))
    nobkd_trn_x, nobkd_trn_y, bkd_trn_x, bkd_trn_y, bkd_x, bkd_y, _, _ = np.load(path, allow_pickle=True)
    FLAGS.batch_size = 500
    FLAGS.epochs = 20
    FLAGS.learning_rate = .8

  return bkd_trn_x, bkd_trn_y, nobkd_trn_x, nobkd_trn_y

def run_backdoor():
  np.random.seed(0)
  
  bkd_trn_x, bkd_trn_y, nobkd_trn_x, nobkd_trn_y = load_data()

  if FLAGS.init_mult<=0.01:
    model = tf.keras.models.load_model(FLAGS.model+'-'+FLAGS.dataset+'-init.h5')
  else:
    model = build_model(bkd_trn_x, bkd_trn_y)
  if FLAGS.backdoor:
    trn_x, trn_y = bkd_trn_x, bkd_trn_y
  else:
    trn_x, trn_y = nobkd_trn_x, nobkd_trn_y

  np.random.seed(None)
  new_seed = np.random.randint(1000000)
  tf.set_random_seed(new_seed)

  train_model(model, trn_x, trn_y, trn_x, trn_y)
  
  # Compute the privacy budget expended.
  eps = compute_epsilon(FLAGS.epochs * trn_x.shape[0] // FLAGS.batch_size,
            FLAGS.batch_size / trn_x.shape[0])
  print('For delta=1e-5, the current epsilon is: %.2f' % eps)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')
  model_dir = auditing_args.args["save_dir"]
  
  if os.path.exists(os.path.join(model_dir, FLAGS.exp_name+'.h5')):
    exit()

  run_backdoor()
  

if __name__ == '__main__':
  app.run(main)
