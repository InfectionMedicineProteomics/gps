class MissingNodeException(Exception):

    def __init__(self, message=''):
        super().__init__(message)

class NonColorOperationException(Exception):

    def __init__(self, message=''):
        super().__init__(message)

class Graph:

    def __init__(self):
        self._nodes = dict()
        self._colors = dict()

    def add_node(self, key, node, color=''):

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


    def add_edge(self, node_from, node_to, weight=0.0, bidirectional=True):
        if node_from not in self._nodes:

            raise MissingNodeException

        elif node_to not in self._nodes:

            raise MissingNodeException

        else:

            self._nodes[node_from].add_edge(
                key=node_to,
                weight=weight
            )

            if bidirectional:
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

    def get_highest_ranking_peakgroups_by_peptide_list(self, node_list, rank):

        nodes_by_rank = list()

        for peptide_key in node_list:

            peptide = self._peptides[peptide_key]

            ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
                rank
            )

            ranked_peakgroup = self._peakgroups[ranked_peakgroup_key]

            nodes_by_rank.append(ranked_peakgroup)

        return nodes_by_rank

    def get_nodes_by_list(self, node_list, rank=0, return_all=False):

        nodes_by_rank = list()

        for peptide_key in node_list:

            peptide = self.get_node(peptide_key)

            if return_all:

                for peakgroup_key in peptide.get_edges():
                    nodes_by_rank.append(
                        self.get_node(peakgroup_key)
                    )

            else:

                if rank > 1:

                    if not peptide.get_num_edges() < 2:
                        ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
                            True,
                            rank
                        )

                else:

                    ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
                        True,
                        rank
                    )

                ranked_peakgroup = self.get_node(ranked_peakgroup_key)

                nodes_by_rank.append(ranked_peakgroup)

        return nodes_by_rank

    def update_node_scores(self, keys, scores, score_name):

        for index, score in zip(keys, scores):

            self[index].scores[score_name] = score
