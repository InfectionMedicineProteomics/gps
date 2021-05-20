

import numpy as np

from sklearn.utils import shuffle

import operator as op

# TODO: THERE IS A BUG WITH THE SCORE FEATURES AND WRITING THE OUTPUT TO SQLITE
class PeakGroup:

    def __init__(self, key='', mz=0.0, rt=0.0, ms2_intensity=0.0, ms1_intensity=0.0):
        self.key = key # feature_id
        self.rt = rt
        self.mz = mz
        self.ms2_intensity = ms2_intensity
        self.ms1_intensity = ms1_intensity
        self.sub_scores = dict()
        self.scores = dict()

    def add_ghost_score_id(self, value):

        self.ghost_score_id = value

    def add_score_column(self, key, value):

        self.scores[key] = value

    def add_sub_score_column(self, key, value):

        self.sub_scores[key] = value

    def get_score_columns(self):

        return [score_column for score_column in self.sub_scores.keys()]

    def get_sub_score_column_array(self, include_score_columns=True):

        score_values = list(self.sub_scores.values())

        if include_score_columns:

            score_values.extend(
                list(self.scores.values())
            )

        return np.asarray(score_values, dtype=np.double)

class Peptide:

    def __init__(self, key='', sequence='', modified_sequence='', charge=0, decoy=0):
        self.key = key
        self.sequence = sequence
        self.modified_sequence = modified_sequence
        self.charge = charge
        self.decoy = decoy
        self.scores = dict()

        if decoy == 0:
            self.target = 1
        else:
            self.target = 0


class Protein:

    def __init__(self, key='', decoy=''):
        self.key = key
        self.protein_accession = key # should change this later to parse
        self.decoy = decoy
        self.scores = dict()


class MissingNodeException(Exception):

    def __init__(self, message=''):
        super().__init__(message)

class NonColorOperationException(Exception):

    def __init__(self, message=''):
        super().__init__(message)


class Node:

    def __init__(self, key, data, color):
        self.key = key
        self.data = data
        self.color = color
        self._edges = dict()

    def add_edge(self, key, weight=0.0):
        self._edges[key] = weight

    def update_weight(self, key, weight):
        self._edges[key] = weight

    def get_edges(self):
        return self._edges.keys()

    def get_edge_by_ranked_weight(self, reverse=True, rank=1):

        weights = [weight for weight in self._edges.values()]

        weights.sort(reverse=reverse)

        rank_index = rank - 1

        try:

            rank_weight = weights[rank_index]

            for key, weight in self._edges.items():

                if rank_weight == weight:
                    return key

        except IndexError:

            return None

    def get_weight(self, key):
        return self._edges[key]


class Graph:

    def __init__(self):
        self._nodes = dict()
        self._colors = dict()

    def add_node(self, key, data, color=''):

        node = Node(
            key=key,
            data=data,
            color=color
        )

        self._nodes[key] = node

        if color:

            if color not in self._colors:
                self._colors[color] = list()

            self._colors[color].append(
                key
            )

        return node

    def get_node(self, key):

        if key in self._nodes:

            return self._nodes[key]

        else:
            print(key)

            raise MissingNodeException(key)

    def __getitem__(self, key):

        if key in self._nodes:

            return self._nodes[key]

    def add_edge(self, node_from, node_to, weight=0.0, directed=True):

        if node_from not in self._nodes:

            raise MissingNodeException

        elif node_to not in self._nodes:

            raise MissingNodeException

        else:

            self._nodes[node_from].add_edge(
                key=node_to,
                weight=weight
            )

            if not directed:

                self._nodes[node_to].add_edge(
                    key=node_from,
                    weight=weight
                )

    def update_edge_weight(self, node_from, node_to, weight=0.0, directed=True):

        if node_from not in self._nodes:

            raise MissingNodeException

        elif node_to not in self._nodes:

            raise MissingNodeException

        else:

            self._nodes[node_from].update_weight(
                key=node_to,
                weight=weight
            )

            if not directed:

                self._nodes[node_to].update_weight(
                    key=node_from,
                    weight=weight
                )

    def get_nodes(self, color=''):

        if color:

            return self._colors[color]

        else:

            return self._nodes.keys()

    def __contains__(self, key):

        return key in self._nodes

    def iter(self, color='', keys=[]):

        if color:

            for key in self._colors[color]:

                yield self._nodes[key]

        elif keys:

            for key in keys:

                yield self._nodes[key]

        else:

            for node in self._nodes.values():

                yield node

    def query_nodes(self, color='', rank=0, return_all=False, query=''):

        comparisons = {
            "<": op.lt,
            "<=": op.le,
            "==": op.eq,
            "!=": op.ne,
            ">": op.gt,
            ">=": op.ge
        }

        if self._colors:

            nodes_by_rank = list()

            for node in self.iter(color=color):

                if return_all:

                    for edge in node.get_edges():

                        nodes_by_rank.append(edge)

                else:
                    ranked_node = node.get_edge_by_ranked_weight(
                        rank=rank
                    )

                    if ranked_node:

                        if query:

                            field_name, operator, comparison_value = query.split(" ")

                            comparison_value = float(comparison_value)

                            node = self.get_node(ranked_node)

                            if field_name in node.data.sub_scores:

                                field_value = node.data.sub_scores[field_name]

                            elif field_name in node.data.scores:

                                field_value = node.data.scores[field_name]

                            if comparisons[operator](field_value, comparison_value):

                                nodes_by_rank.append(
                                    ranked_node
                                )

                        else:

                            nodes_by_rank.append(
                                ranked_node
                            )

        else:

            raise NonColorOperationException(
                message="Cannot perform partite operation on graph with no partite elements."
            )

        return nodes_by_rank

    def get_ranked_nodes_by_node_list(self, node_list=[], rank=0, return_all=False):

        nodes_by_rank = list()

        for node in self.iter(keys=node_list):

            if return_all:

                for edge in node.get_edges():
                    nodes_by_rank.append(edge)

            else:
                ranked_node = node.get_edge_by_ranked_weight(
                    rank=rank
                )

                if ranked_node:
                    nodes_by_rank.append(
                        ranked_node
                    )

        return nodes_by_rank



def parse_scores_labels_index(graph, node_keys, is_decoy=False, include_score_columns=False):

    scores = list()
    score_labels = list()
    score_indices = list()

    for idx, node_key in enumerate(node_keys):

        node = graph.get_node(node_key)

        score_array = node.data.get_sub_score_column_array(
            include_score_columns=include_score_columns
        )

        scores.append(score_array)

        if is_decoy:

            score_labels.append(
                [0.0]
            )

        else:

            score_labels.append(
                [1.0]
            )

        score_indices.append(
            node.data.key
        )

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)
    score_indices = np.array(score_indices, dtype=np.str)

    return scores, score_labels, score_indices


def get_training_peptides(peptide_folds, fold_num):

    training_peptides = list()

    for training_fold_idx, peptide_ids in enumerate(peptide_folds):

        if training_fold_idx != fold_num:

            for peptide_id in peptide_ids:

                training_peptides.append(peptide_id)

    return training_peptides


def preprocess_training_data(graph, training_peptides):
    training_top_ranked = graph.get_ranked_nodes_by_node_list(
        node_list=training_peptides,
        rank=1
    )

    training_second_ranked = graph.get_ranked_nodes_by_node_list(
        node_list=training_peptides,
        rank=2
    )

    target_scores, target_labels, target_indices = parse_scores_labels_index(
        graph=graph,
        node_keys=training_top_ranked,
        is_decoy=False
    )

    false_target_scores, false_target_labels, false_target_indices = parse_scores_labels_index(
        graph=graph,
        node_keys=training_second_ranked,
        is_decoy=True
    )

    scores = np.concatenate(
        [
            target_scores,
            false_target_scores
        ]
    )

    score_labels = np.concatenate(
        [
            target_labels,
            false_target_labels
        ]
    )

    score_indices = np.concatenate(
        [
            target_indices,
            false_target_indices
        ]
    )

    train_data, train_labels, train_indices = shuffle(
        scores, score_labels, score_indices,
        random_state=42
    )

    return train_data, train_labels, train_indices


def preprocess_data_to_score(graph, node_list, rank=0, return_all=False):

    if return_all:

        peakgroup_keys = graph.get_ranked_nodes_by_node_list(
            node_list=node_list,
            return_all=True
        )

    else:

        peakgroup_keys = graph.get_ranked_nodes_by_node_list(
            node_list=node_list,
            rank=rank
        )

    scores, labels, indices = parse_scores_labels_index(
        graph=graph,
        node_keys=peakgroup_keys,
        is_decoy=False
    )

    return scores, labels, indices


def update_peakgroup_scores(graph, testing_indices, testing_scores, score_name):

    for index, score in zip(testing_indices, testing_scores):

        graph[index].data.scores[score_name] = score


def get_score_array(graph, node_list, score_column=''):

    score_array = list()

    for node in graph.iter(keys=node_list):

        score = node.data.scores[score_column]

        score_array.append(score)

    return np.array(score_array)


def calc_score_grouped_by_level(graph, function=None, level='', score_column='', new_column_name=''):

    for node in graph.iter(color=level):

        scores = list()

        for key in node.get_edges():

            scores.append(
                graph[key].data.scores[score_column]
            )

        scores = np.array(scores, dtype=np.float64)

        score = function(scores)

        node.data.scores[new_column_name] = score
