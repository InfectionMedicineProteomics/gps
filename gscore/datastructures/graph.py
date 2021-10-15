import operator

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


    def scale_peakgroup_retention_times(self):

        self.min_rt_val = 0.0
        self.max_rt_val = 0.0

        a, b = 0.0, 100.0

        for peakgroup_id in self._colors['peakgroup']:

            peakgroup = self.get_node(peakgroup_id)

            if peakgroup.end_rt > self.max_rt_val:

                self.max_rt_val = peakgroup.end_rt

        for peakgroup_id in self._colors['peakgroup']:

            peakgroup = self.get_node(peakgroup_id)

            peakgroup.scaled_rt_start = a + (
                ((peakgroup.start_rt - self.min_rt_val) * (b - a)) / (self.max_rt_val - self.min_rt_val)
            )

            peakgroup.scaled_rt_apex = a + (
                    ((peakgroup.rt - self.min_rt_val) * (b - a)) / (self.max_rt_val - self.min_rt_val)
            )

            peakgroup.scaled_rt_end = a + (
                    ((peakgroup.end_rt - self.min_rt_val) * (b - a)) / (self.max_rt_val - self.min_rt_val)
            )

    def get_ranked_peakgroups(self, rank: int = 1, target: int = 1):

        ranked_peakgroups = list()

        for node in self._colors['peptide']:

            peptide_node = self._nodes[node]

            if peptide_node.target == target:

                ranked_peakgroup_key = peptide_node.get_edge_by_ranked_weight(rank=rank)

                ranked_peakgroup = self._nodes[ranked_peakgroup_key]

                ranked_peakgroups.append(ranked_peakgroup)

        return ranked_peakgroups


    def filter_ranked_peakgroups(self, rank: int = 1, score_column: str = '', value: float = 0.0, user_operator: str = '', target: int = 1):

        ranked_peakgroups = list()

        operators = {
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge
        }

        for node in self._colors['peptide']:

            peptide_node = self._nodes[node]

            if peptide_node.target == target:

                ranked_peakgroup_key = peptide_node.get_edge_by_ranked_weight(rank=rank)

                ranked_peakgroup = self._nodes[ranked_peakgroup_key]

                score = ranked_peakgroup.scores[score_column]

                ## TODO: this is only because some probabilities are None, need to fix this
                if score:
                    if operators[user_operator](score, value):

                        ranked_peakgroups.append(ranked_peakgroup)

        return ranked_peakgroups

    def get_peptide_nodes(self):

        peptides = []

        for peptide_key in self._colors['peptide']:

            peptide = self._nodes[peptide_key]

            peptides.append(peptide)

        return peptides

    def calculate_global_level_scores(self, function = None, level = '', score_column = '', new_column_name = ''):

        for node_key in self._colors[level]:

            node = self._nodes[node_key]

            scores = list()

            for key in node.get_edges():

                child_node = self._nodes[key]

                if level == "peptide":

                    if child_node.color == "peakgroup":

                        scores.append(
                            self._nodes[key].scores[score_column]
                        )

                elif level == "protein":

                    if child_node.color == "peptide":

                        scores.append(
                            self._nodes[key].scores[score_column]
                        )


            if scores:

                score = function(scores)

            else:

                score = -20.0

            node.scores[score_column] = score
