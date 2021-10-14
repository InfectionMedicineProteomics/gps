import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pomegranate import (
    GeneralMixtureModel,
    NormalDistribution
)

from gscore import distributions
from gscore import peakgroups
from gscore.parsers import osw, queries

from gscore.distributions import LabelDistribution, ScoreDistribution


def plot_distributions(
        target_distribution,
        null_distribution,
        x_axis,
        all_targets_distribution,
        fig_path
):

    fig, ax = plt.subplots()
    sns.lineplot(
        x=x_axis,
        y=target_distribution,
        ax=ax,
        label='Targets'
    )
    sns.lineplot(
        x=x_axis,
        y=null_distribution,
        ax=ax,
        label='False Targets'
    )
    plt.savefig(fig_path)


def get_target_and_decoy_scores(input_graphs, level):

    target_scores = dict()
    decoy_scores = dict()

    for graph in input_graphs:

        for key in graph.get_nodes(level):

            node = graph[key]

            if node.target == 1:

                if key not in target_scores:

                    target_scores[key] = node.scores['d_score']

                if node.scores["d_score"] > target_scores[key]:

                    target_scores[key] = node.scores['d_score']
            else:

                if key not in decoy_scores:

                    decoy_scores[key] = node.scores['d_score']

                if node.scores["d_score"] > decoy_scores[key]:

                    decoy_scores[key] = node.scores['d_score']

    return target_scores, decoy_scores


def fit_distributions(target_scores, decoy_scores):

    combined_scores = np.concatenate([target_scores, decoy_scores])

    axis_min = combined_scores.min()
    axis_max = combined_scores.max()

    x_plot = np.linspace(
        start=axis_min - 3,
        stop=axis_max + 3,
        num=1000
    )[:, np.newaxis]

    print("Fitting Distributions.")

    target_distribution = LabelDistribution(
        axis_span=(x_plot.min() - 3, x_plot.max() + 3)
    )
    target_distribution.fit(
        np.array(target_scores).reshape(-1, 1)
    )

    decoy_distribution = LabelDistribution(
        axis_span=(x_plot.min() - 3, x_plot.max() + 3)
    )
    decoy_distribution.fit(
        np.array(decoy_scores).reshape(-1, 1)
    )

    score_distribution = ScoreDistribution(
        decoy_scores=decoy_distribution.values(x_plot),
        target_scores=target_distribution.values(x_plot),
        x_axis=x_plot
    )

    return score_distribution, target_distribution, decoy_distribution, x_plot


def main(args):

    print(f'Building q-value scoring models')

    if args.use_decoys:

        pass

    else:

        osw_query = queries.SelectPeakGroups.FETCH_SCORED_DATA_DECOY_FREE

    input_graphs = []

    print(len(args.input_files))

    for input_file in args.input_files:

        print(f"Parsing input file {input_file}")

        graph, _ = osw.fetch_peakgroup_graph(
            osw_path=input_file,
            query=queries.SelectPeakGroups.FETCH_ALL_SCORED_DATA
        )

        graph.calculate_global_level_scores(
            function=np.max,
            level="peptide",
            score_column="d_score",
            new_column_name="d_score"
        )

        graph.calculate_global_level_scores(
            function=np.max,
            level="protein",
            score_column="d_score",
            new_column_name="d_score"
        )

        input_graphs.append(graph)

    peakgroup_target_scores, peakgroup_decoy_scores = get_target_and_decoy_scores(input_graphs, "peakgroup")

    peptide_target_scores, peptide_decoy_scores = get_target_and_decoy_scores(input_graphs, "peptide")

    protein_target_scores, protein_decoy_scores = get_target_and_decoy_scores(input_graphs, "protein")

    peakgroup_score_distribution, peakgroup_target_distribution, peakgroup_decoy_destribution, peakgroup_x_plot = fit_distributions(
        list(peakgroup_target_scores.values()),
        list(peakgroup_decoy_scores.values())
    )

    peptide_score_distribution, peptide_target_distribution, peptide_decoy_destribution, peptide_x_plot = fit_distributions(
        list(peptide_target_scores.values()), list(peptide_decoy_scores.values()))

    protein_score_distribution, protein_target_distribution, protein_decoy_destribution, protein_x_plot = fit_distributions(
        list(protein_target_scores.values()), list(protein_decoy_scores.values()))

    with open(args.peptide_model_output, 'wb') as pkl:
        pickle.dump(peptide_score_distribution, pkl)

    with open(args.protein_model_output, 'wb') as pkl:
        pickle.dump(protein_score_distribution, pkl)

    with open(args.peakgroup_model_output, 'wb') as pkl:
        pickle.dump(peakgroup_score_distribution, pkl)
