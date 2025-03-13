from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler

from DFIGCN_LP.optimizer import OptimizerAE, OptimizerVAE
from DFIGCN_LP.input_data import load_data
from DFIGCN_LP.model import GCNModelAE, DFIGCNModelAE, GCNModelVAE
from DFIGCN_LP.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
from DFIGCN_LP.utils import preprocess_bilinear, load_mat_data, preprocess_features



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'dfigcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'conflict', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

flags.DEFINE_float('alpha', 0.4, 'alpha balancing BGCN')
flags.DEFINE_float('beta', 0.6, 'beta balancing BGCN-target')


def train_process():

    tf.reset_default_graph()

    model_str = FLAGS.model
    dataset_str = FLAGS.dataset

    # Load data
    if dataset_str =='powergrid':
        adj, features = load_mat_data(dataset_str)

    elif dataset_str in ['protein', 'metabolic', 'conflict']:
        adj, features = load_mat_data(dataset_str)
        if dataset_str == 'protein':
            negatives = features < 0.0
            r, c, values = sp.find(negatives)
            features[r, c] = 0.0
        else:
            features = features.toarray()
            features = MinMaxScaler().fit_transform(features)
            features = sp.csr_matrix(features)

    else:
        adj, features = load_data(dataset_str)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train
    if dataset_str != 'conflict':
        adj.setdiag(1.0)

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless


    # Some preprocessing
    # adj_norm = preprocess_graph(adj)
    A_2,adj_hop1_all, adj_hop2_all, adj_hop1_neig, adj_hop2_neig, N_hop1_target, N_hop2_target = preprocess_bilinear(adj)
    support = [A_2, adj_hop1_all, adj_hop2_all, adj_hop1_neig, adj_hop2_neig, N_hop1_target, N_hop2_target]
    num_supports = 7

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    if model_str == 'dfigcn_ae':
        model = DFIGCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae' or 'dfigcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cost_val = []
    acc_val = []


    def get_roc_score(edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score


    cost_val = []
    acc_val = []
    val_roc_score = []

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(support, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
        val_roc_score.append(roc_curr)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    return roc_score, ap_score

if __name__ == "__main__":

    rocs = []
    aps = []
    for i in range(10):
        roc, ap = train_process()
        rocs.append((roc))
        aps.append((ap))

    mean_roc = np.mean(rocs)
    mean_ap = np.mean(aps)
    print('mean roc score:', mean_roc, ' mean ap score:', mean_ap)

