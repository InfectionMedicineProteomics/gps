import numba
from numba import types, typed
from numba.experimental import jitclass

from gscore.peakgroups import (
    Protein,
    Peptide,
    PeakGroup
)

edge_kv_types = (types.unicode_type, types.float64)

protein_kv_types = (types.unicode_type, numba.extending.as_numba_type(Protein))
peptide_kv_types = (types.unicode_type, numba.extending.as_numba_type(Peptide))
peakgroup_kv_types = (types.unicode_type, numba.extending.as_numba_type(PeakGroup))
node_kv_types = (types.unicode_type, types.unicode_type)
colors_kv_types = (types.unicode_type, types.ListType(types.unicode_type))

from numba import jit

graph_spec = [
    ('_nodes', types.DictType(*node_kv_types)),
    ('_colors', types.DictType(*colors_kv_types)),
    ('_proteins', types.DictType(*protein_kv_types)),
    ('_peptides', types.DictType(*peptide_kv_types)),
    ('_peakgroups', types.DictType(*peakgroup_kv_types)),
]


@jitclass(graph_spec)
class Graph:

    def __init__(self):

        self._nodes = typed.Dict.empty(*node_kv_types)
        self._colors = typed.Dict.empty(*colors_kv_types)
        self._proteins = typed.Dict.empty(*protein_kv_types)
        self._peptides = typed.Dict.empty(*peptide_kv_types)
        self._peakgroups = typed.Dict.empty(*peakgroup_kv_types)

    def add_protein(self, key, protein):

        self._proteins[key] = protein

        self._nodes[key] = "protein"

        if 'protein' not in self._colors:
            self._colors['protein'] = typed.List.empty_list(types.unicode_type)

        self._colors['protein'].append(
            key
        )

    def add_peptide(self, key, peptide):

        self._peptides[key] = peptide

        self._nodes[key] = "peptide"

        if 'peptide' not in self._colors:
            self._colors['peptide'] = typed.List.empty_list(types.unicode_type)

        self._colors['peptide'].append(
            key
        )

    def add_peakgroup(self, key, peakgroup):

        self._peakgroups[key] = peakgroup

        self._nodes[key] = "peakgroup"

        if 'peakgroup' not in self._colors:
            self._colors['peakgroup'] = typed.List.empty_list(types.unicode_type)

        self._colors['peakgroup'].append(
            key
        )

    def get_peakgroup(self, key):

        return self._peakgroups[key]

    def get_peptide(self, key):

        return self._peptides[key]

    def get_protein(self, key):

        return self._proteins[key]

    def add_edge(self, node_from, node_to, weight=0.0, bidirectional=False):

        if node_from not in self._nodes:

            raise Exception("Missing node")

        elif node_to not in self._nodes:

            raise Exception("Missing node")

        else:

            node_from_color = self._nodes[node_from]
            node_to_color = self._nodes[node_to]

            if node_from_color == "protein":

                self._proteins[node_from].add_edge(
                    key=node_to,
                    weight=weight
                )

            elif node_from_color == "peptide":

                self._peptides[node_from].add_edge(
                    key=node_to,
                    weight=weight
                )

            elif node_from_color == "peakgroup":

                self._peakgroups[node_from].add_edge(
                    key=node_to,
                    weight=weight
                )

            if bidirectional:

                if node_to_color == "protein":

                    self._proteins[node_to].add_edge(
                        key=node_from,
                        weight=weight
                    )

                elif node_to_color == "peptide":

                    self._peptides[node_to].add_edge(
                        key=node_from,
                        weight=weight
                    )

                elif node_to_color == "peakgroup":

                    self._peakgroups[node_to].add_edge(
                        key=node_from,
                        weight=weight
                    )

    def update_peakgroup_scores(self, testing_indices, testing_scores, score_name):

        counter = 0

        for key in testing_indices:

            peakgroup = self._peakgroups[key]

            peakgroup.scores[score_name] = testing_scores[counter]

            counter += 1


    def update_edge_weight(self, node_from, node_to, weight=0.0, bidirectional=False):

        if node_from not in self._nodes:

            raise Exception("Missing node")

        elif node_to not in self._nodes:

            raise Exception("Missing node")

        else:

            node_from_color = self._nodes[node_from]
            node_to_color = self._nodes[node_to]

            if node_from_color == "protein":

                self._proteins[node_from].update_weight(
                    key=node_to,
                    weight=weight
                )

            elif node_from_color == "peptide":

                self._peptides[node_from].update_weight(
                    key=node_to,
                    weight=weight
                )

            elif node_from_color == "peakgroup":

                self._peakgroups[node_from].update_weight(
                    key=node_to,
                    weight=weight
                )

            if bidirectional:

                if node_to_color == "protein":

                    self._proteins[node_to].update_weight(
                        key=node_from,
                        weight=weight
                    )

                elif node_to_color == "peptide":

                    self._peptides[node_to].update_weight(
                        key=node_from,
                        weight=weight
                    )

                elif node_to_color == "peakgroup":

                    self._peakgroups[node_to].update_weight(
                        key=node_from,
                        weight=weight
                    )

    def get_nodes(self, color=''):

        return_nodes = typed.List()

        if color:

            return_nodes = self._colors[color]

        else:

            for key in self._nodes.keys():
                return_nodes.append(key)

        return return_nodes

    def contains(self, key):

        return key in self._nodes

    def get_highest_ranking_peakgroups_by_peptide_list(self, node_list, rank):

        nodes_by_rank = typed.List()

        for peptide_key in node_list:

            peptide = self._peptides[peptide_key]

            ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
                rank
            )

            ranked_peakgroup = self._peakgroups[ranked_peakgroup_key]

            nodes_by_rank.append(ranked_peakgroup)

        return nodes_by_rank
