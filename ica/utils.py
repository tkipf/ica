"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
import networkx as nx
import cPickle as pkl
import numpy as np
import scipy.sparse as sp
import os.path
from ica.graph import UndirectedGraph, Node, Edge
from ica.aggregators import Count, Prop


def build_graph(adj, features, labels):
    edges = np.array(adj.nonzero()).T
    y_values = np.array(labels.nonzero()).T

    domain_labels = []
    for i in range(labels.shape[1]):
        domain_labels.append("c" + str(i))

    # create graph
    graph = UndirectedGraph()
    id_obj_map = []
    for i in range(adj.shape[0]):
        n = Node(i, features[i, :], domain_labels[y_values[i, 1]])
        graph.add_node(n)
        id_obj_map.append(n)
    for e in edges:
        graph.add_edge(Edge(id_obj_map[e[1]], id_obj_map[e[0]]))

    return graph, domain_labels


def pick_aggregator(agg, domain_labels):
    if agg == 'count':
        aggregator = Count(domain_labels)
    elif agg == 'prop':
        aggregator = Prop(domain_labels)
    else:
        raise ValueError('Invalid argument for agg (aggregation operator): ' + str(agg))
    return aggregator


def create_map(graph, train_indices):
    conditional_map = {}
    for i in train_indices:
        conditional_map[graph.node_list[i]] = graph.node_list[i].label
    return conditional_map


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1] - 1))
        ty_extended_ = np.ones((len(test_idx_range_full), 1))  # add dummy labels
        ty_extended = np.hstack([ty_extended, ty_extended_])
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    return adj, features, labels, idx_train, idx_val, idx_test
