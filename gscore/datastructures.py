import numpy as np

# TODO: THERE IS A BUG WITH THE SCORE FEATURES AND WRITING THE OUTPUT TO SQLITE
class PeakGroup:

    def __init__(self, key='', mz=0.0, rt=0.0, ms2_intensity=0.0, ms1_intensity=0.0):
        self.key = key # feature_id
        self.rt = rt
        self.mz = mz
        self.ms2_intensity = ms2_intensity
        self.ms1_intensity = ms1_intensity
        self.sub_scores = dict()

    def add_sub_score_column(self, key, value):

        self.sub_scores[key] = value

    def get_score_columns(self):

        return [score_column for score_column in self.sub_scores.keys()]

    def get_score_column_array(self):

        return np.asarray(list(self.sub_scores.values()), dtype=np.double)

class Peptide:

    def __init__(self, key='', sequence='', modified_sequence='', charge=0, decoy=0):
        self.key = key
        self.sequence = sequence
        self.modified_sequence = modified_sequence
        self.charge = charge
        self.decoy = decoy

        if decoy == 0:
            self.target = 1
        else:
            self.target = 0


class Protein:

    def __init__(self, key='', decoy=''):
        self.key = key
        self.protein_accession = key # should change this later to parse
        self.decoy = decoy


class MissingNodeException(Exception):

    pass

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
