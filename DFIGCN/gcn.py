#!/usr/local/bin/python
from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
from DFIGCN.utils import *
from DFIGCN.models import DFIGCN


# Set random seed
# seed = 123
# np.random.seed(seed)
# tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_string('model', 'dfigcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('patience', 100, 'early stop #epoch')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('alpha', 0.3, 'alpha balancing DFIGCN')
flags.DEFINE_float('beta', 0.7, 'beta balancing DFIGCN-target')

checkpt_file = 'pre_trained/mod_'+FLAGS.dataset+'.ckpt'
# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

#bilinear matrices
A_2,adj_hop1_all, adj_hop2_all, adj_hop1_neig, adj_hop2_neig, N_hop1_target, N_hop2_target = preprocess_bilinear(adj)
# Some preprocessing
features= preprocess_features(features)

if FLAGS.model == 'dfigcn':
    support = [A_2, adj_hop1_all, adj_hop2_all, adj_hop1_neig, adj_hop2_neig, N_hop1_target, N_hop2_target]
    num_supports = 7
    model_func = DFIGCN


r_time1 = time.time()
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(y_train.shape[0], y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout

}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.cross_entropy_loss], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())


vacc_mx = 0.0
vlss_mn = np.inf
curr_step = 0
saver = tf.train.Saver()

# Train model
for epoch in range(FLAGS.epochs):#FLAGS.epochs

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.cross_entropy_loss], feed_dict=feed_dict)

    # Validation
    cost, val_acc, val_loss, duration = evaluate(features, support, y_val, val_mask, placeholders)

    # Testing
    test_cost, test_acc, tst_loss, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(outs[3]),"train_acc=", "{:.4f}".format(outs[2]),
          "val_loss=", "{:.5f}".format(val_loss),"val_acc=", "{:.4f}".format(val_acc),
          "tst_loss=", "{:.5f}".format(tst_loss), "tst_acc=", "{:.4f}".format(test_acc),
          "time=","{:.3f}".format(time.time()-t))

    if val_acc >= vacc_mx or val_loss <= vlss_mn:
        if val_acc >= vacc_mx and val_loss <= vlss_mn:
            vacc_early = val_acc
            vlss_early = val_loss
            epoch_early = epoch+1
            saver.save(sess, checkpt_file)
        vacc_mx = np.max((val_acc,vacc_mx))
        vlss_mn = np.min((val_loss, vlss_mn))
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == FLAGS.patience:
            print('Early stop!')
            break

saver.restore(sess, checkpt_file)
# Testing
test_cost, test_acc, tst_loss, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("test_loss=", "{:.5f}".format(tst_loss), "test_acc=", "{:.4f}".format(test_acc))
sess.close()
r_time2 = time.time()
print('early stop #epoch', epoch_early, 'val_loss=', "{:.5f}".format(vlss_early), 'val_acc=', "{:.4f}".format(vacc_early))


