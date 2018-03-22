from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN
import random

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('train_num', 90, 'we simply define a train_num to split data into train/test dataset')

# Load data
adjs, features, labels = load_data()
assert((len(adjs)==len(features)) and (len(adjs)==len(labels)))
full_indices = list(range(len(labels)))
random.shuffle(full_indices)
train_indices = full_indices[:FLAGS.train_num]
test_indices = full_indices[FLAGS.train_num:]

#preprocess use gcn
print('preprocessing...')
supports = [[preprocess_adj(adj)] for adj in adjs]  #add a [] outside adj TO fit original data format.
features = [preprocess_features(feature) for feature in features]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[0][2], dtype=tf.int64)),   #since we have many features, use any of them calculate shape.
    'labels': tf.placeholder(tf.float32, shape=(None, 2)),  #2分类问题
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[0][2][1], logging=True)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, placeholders)
    outs_val = sess.run([model.loss], feed_dict=feed_dict_val)
    return outs_val[0], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    for index in train_indices:
        print('index:'+str(index))
        feature = features[index]
        support = supports[index]
        label = labels[index]
        if (label==0):
            y_train = np.array([[0, 1]])
        else:
            y_train = np.array([[1, 0]])
        # Construct feed dictionary
        feed_dict = construct_feed_dict(feature, support, y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

        # Validation
        cost_val.append(outs[1])

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cost_val[-1]),
          "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

