"""" This implementation is largely based on and adapted from:
 https://github.com/sskhandle/Iterative-Classification """
import numpy as np
import scipy.sparse as sp


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    md = __import__(module)
    for comp in parts[1:]:
        md = getattr(md, comp)
    return md


class Classifier(object):
    def __init__(self, scikit_classifier_name, **classifier_args):
        classifer_class = get_class(scikit_classifier_name)
        self.clf = classifer_class(**classifier_args)

    def fit(self, graph, train_indices):
        raise NotImplementedError

    def predict(self, graph, test_indices, conditional_node_to_label_map=None):
        raise NotImplementedError


class LocalClassifier(Classifier):
    def fit(self, graph, train_indices):

        feature_list = []
        label_list = []
        g = graph
        n = g.node_list
        training_nodes = [n[i] for i in train_indices]

        for nodes in training_nodes:
            feature_list.append(nodes.feature_vector)
            label_list.append(nodes.label)

        feature_list = sp.vstack(feature_list)
        feature_list = sp.csr_matrix(feature_list, dtype=np.float64)

        self.clf.fit(feature_list, label_list)
        return

    def predict(self, graph, test_indices, conditional_node_to_label_map=None):

        feature_list = []
        g = graph
        n = g.node_list
        testing_nodes = [n[i] for i in test_indices]

        for nodes in testing_nodes:
            feature_list.append(nodes.feature_vector)

        feature_list = sp.vstack(feature_list)
        feature_list = sp.csr_matrix(feature_list, dtype=np.float64)

        y = self.clf.predict(feature_list)
        return y


class RelationalClassifier(Classifier):
    def __init__(self, scikit_classifier_name, aggregator, **classifier_args):
        super(RelationalClassifier, self).__init__(scikit_classifier_name, **classifier_args)
        self.aggregator = aggregator

    def fit(self, graph, train_indices, local_classifier, bootstrap):
        conditional_map = {}

        if bootstrap:
            predictclf = local_classifier.predict(graph, range(len(graph.node_list)))
            conditional_map = self.cond_mp_upd(graph, conditional_map, predictclf, range(len(graph.node_list)))

        for i in train_indices:
            conditional_map[graph.node_list[i]] = graph.node_list[i].label
        features = []
        aggregates = []
        labels = []
        for i in train_indices:
            features.append(graph.node_list[i].feature_vector)
            labels.append(graph.node_list[i].label)
            aggregates.append(sp.csr_matrix(self.aggregator.aggregate(graph,
                                                                      graph.node_list[i],
                                                                      conditional_map), dtype=np.float64))
        features = sp.vstack(features)
        features = sp.csr_matrix(features, dtype=np.float64)
        aggregates = sp.vstack(aggregates)
        features = sp.hstack([features, aggregates])

        self.clf.fit(features, labels)

    def predict(self, graph, test_indices, conditional_map=None):
        features = []
        aggregates = []

        for i in test_indices:
            features.append(graph.node_list[i].feature_vector)
            aggregates.append(sp.csr_matrix(self.aggregator.aggregate(graph,
                                                                      graph.node_list[i],
                                                                      conditional_map), dtype=np.float64))
        features = sp.vstack(features)
        features = sp.csr_matrix(features, dtype=np.float64)
        aggregates = sp.vstack(aggregates)
        features = sp.hstack([features, aggregates])

        return self.clf.predict(features)

    def cond_mp_upd(self, graph, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[graph.node_list[indices[x]]] = pred[x]
        return conditional_map


class ICA(Classifier):
    def __init__(self, local_classifier, relational_classifier, bootstrap, max_iteration=10):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.bootstrap = bootstrap
        self.max_iteration = max_iteration

    def fit(self, graph, train_indices):
        self.local_classifier.fit(graph, train_indices)
        self.relational_classifier.fit(graph, train_indices, self.local_classifier, self.bootstrap)

    def predict(self, graph, eval_indices, test_indices, conditional_node_to_label_map=None):
        predictclf = self.local_classifier.predict(graph, eval_indices)
        conditional_node_to_label_map = self.cond_mp_upd(graph,
                                                         conditional_node_to_label_map,
                                                         predictclf, eval_indices)

        relation_predict = []
        temp = []
        for iter in range(self.max_iteration):
            for x in eval_indices:
                temp.append(x)
                rltn_pred = list(self.relational_classifier.predict(graph, temp, conditional_node_to_label_map))
                conditional_node_to_label_map = self.cond_mp_upd(graph, conditional_node_to_label_map, rltn_pred, temp)
                temp.remove(x)
        for ti in test_indices:
            relation_predict.append(conditional_node_to_label_map[graph.node_list[ti]])
        return relation_predict

    def cond_mp_upd(self, graph, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[graph.node_list[indices[x]]] = pred[x]
        return conditional_map
