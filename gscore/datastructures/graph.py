import operator as op

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

    def iter_edges(self, graph):

        for key, weight in self._edges.items():

            yield graph[key]

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