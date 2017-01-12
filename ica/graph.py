"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
from collections import defaultdict


class Graph(object):
    def __init__(self):
        self.node_list = []

    def add_node(self, n):
        self.node_list.append(n)

    def add_edge(self, e):
        raise NotImplementedError

    def get_neighbors(self, n):
        raise NotImplementedError


class Node(object):
    def __init__(self, node_id, feature_vector=None, label=None):
        self.node_id = node_id
        self.feature_vector = feature_vector
        self.label = label


class Edge(object):
    def __init__(self, from_node, to_node, feature_vector=None, label=None):
        self.from_node = from_node
        self.to_node = to_node
        self.feature_vector = feature_vector
        self.label = label


class UndirectedGraph(Graph):
    def __init__(self):
        super(UndirectedGraph, self).__init__()
        self.neighbors = defaultdict(set)

    def add_edge(self, e):
        self.neighbors[e.from_node].add(e.to_node)
        self.neighbors[e.to_node].add(e.from_node)

    def get_neighbors(self, n):
        return self.neighbors[n]
