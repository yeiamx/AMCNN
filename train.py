from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adjs, features, labels = load_data()
assert((len(adjs)==len(features)) and (len(adjs)==len(labels)))

#preprocess use gcn
print('preprocessing...')
supports = [[preprocess_adj(adj)] for adj in adjs]  #add a [] outside adj TO fit original data format.
features = [preprocess_features(feature) for feature in features]
num_supports = 1
model_func = GCN

#pprint.pprint(support)
#pprint.pprint(feature)

# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(feature, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(feature, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

cost_val = []

# Train model
for index in range(len(supports)):
    support = supports[index]
    feature = features[index]
    y_train = labels[index]
    train_mask = np.array(list(range(len(y_train))))
    train_mask[:] = True

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[index][2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[index][2][1], logging=True)

    # Init variables
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        feed_dict = construct_feed_dict(feature, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Print results
        print("Index:", '%04d' % (index + 1), "Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]))

        # Validation
        # cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        # cost_val.append(cost)

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
#test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
#print("Test set results:", "cost=", "{:.5f}".format(test_cost),
#      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

