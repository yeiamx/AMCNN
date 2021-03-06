import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import pprint
import os
from dataIO import *

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data():
    """Load data."""
    print('loading data...')
    lil_adjancency_matrix_list = []
    lil_features_matrix_list = []
    label = []

    chair_path_list = os.listdir(chair_file_path)
    bathhub_path_list = os.listdir(bathtub_file_path1)
    index = 0
    total_len = len(chair_path_list)+len(bathhub_path_list)

    print('processing chairs...')
    for file_name in chair_path_list:
        if ('am' in file_name.split('.')[-2]):
            am = np.load(chair_file_path+'/'+file_name)
            lil_am = sp.lil_matrix(am)
            lil_adjancency_matrix_list.append(lil_am)
        if ('feature' in file_name.split('.')[-2]):
            feature = np.load(chair_file_path+'/'+file_name)
            lil_feature = sp.lil_matrix(feature)
            lil_features_matrix_list.append(lil_feature)

            label.append(0)
            index+=1
            #print('get '+str(index)+' in '+str(total_len))

    print('processing bathtubs...')
    for file_name in bathhub_path_list:
        if ('am' in file_name.split('.')[-2]):
            am = np.load(bathtub_file_path1+'/'+file_name)
            lil_am = sp.lil_matrix(am)
            lil_adjancency_matrix_list.append(lil_am)
        if ('feature' in file_name.split('.')[-2]):
            feature = np.load(bathtub_file_path1+'/'+file_name)
            lil_feature = sp.lil_matrix(feature)
            lil_features_matrix_list.append(lil_feature)

            label.append(1)
            index+=1
            #print('get '+str(index)+' in '+str(total_len))

    return lil_adjancency_matrix_list, lil_features_matrix_list, label

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[0][1].shape})
    return feed_dict
