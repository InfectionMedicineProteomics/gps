import random

from sklearn.utils import shuffle

import numpy as np

def get_peptide_id_folds(graph, num_folds):

    peptide_ids = list(graph.get_nodes('peptide'))

    random.seed(42)
    random.shuffle(peptide_ids)

    peptide_folds = np.array_split(
        peptide_ids,
        num_folds
    )

    return peptide_folds


def get_training_peptides(peptide_folds, fold_num):

    training_peptides = list()

    for training_fold_idx, peptide_ids in enumerate(peptide_folds):

        if training_fold_idx != fold_num:

            for peptide_id in peptide_ids:

                training_peptides.append(peptide_id)

    return training_peptides


def reformat_data(peakgroups, include_score_columns=False):

    scores = list()
    score_labels = list()
    score_indices = list()

    for idx, peakgroup in enumerate(peakgroups):

        score_array = peakgroup.get_sub_score_column_array(
            include_score_columns
        )

        scores.append(score_array)

        score_labels.append(
            [peakgroup.target]
        )

        score_indices.append(
            peakgroup.key
        )

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)
    score_indices = np.array(score_indices, dtype=np.str)

    return scores, score_labels, score_indices


# def get_peakgroups(graph, node_list, rank=0, return_all=False):
#
#     nodes_by_rank = list()
#
#     for peptide_key in node_list:
#
#         peptide = graph.get_node(peptide_key)
#
#         if return_all:
#
#             for peakgroup_key in peptide.get_edges():
#
#                 nodes_by_rank.append(
#                     graph.get_node(peakgroup_key)
#                 )
#
#         else:
#
#             if rank > 1:
#
#                 if not peptide.get_num_edges() < 2:
#
#
#
#                     ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
#                         True,
#                         rank
#                     )
#
#             else:
#
#                 ranked_peakgroup_key = peptide.get_edge_by_ranked_weight(
#                     True,
#                     rank
#                 )
#
#             ranked_peakgroup = graph.get_node(ranked_peakgroup_key)
#
#             nodes_by_rank.append(ranked_peakgroup)
#
#     return nodes_by_rank


def preprocess_data(graph, node_list, return_all=False, use_decoys=False):

    peakgroups = list()

    if return_all:

        all_peakgroups = graph.get_nodes_by_list(
            node_list=node_list,
            return_all=True
        )

        peakgroups.extend(all_peakgroups)

    else:

        training_top_ranked = graph.get_nodes_by_list(
            node_list=node_list,
            rank=1
        )

        peakgroups.extend(training_top_ranked)

        if not use_decoys:

            training_second_ranked = graph.get_nodes_by_list(
                node_list=node_list,
                rank=2
            )

            for peakgroup_node in training_second_ranked:

                peakgroup_node.target = 0.0
                peakgroup_node.decoy = 1.0

                peakgroups.append(peakgroup_node)

    peakgroup_scores, peakgroup_labels, peakgroup_keys = reformat_data(
        peakgroups=peakgroups
    )


    peakgroup_scores, peakgroup_labels, peakgroup_keys = shuffle(
        peakgroup_scores,
        peakgroup_labels,
        peakgroup_keys,
        random_state=42
    )

    return peakgroup_scores, peakgroup_labels, peakgroup_keys
