import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from gscore import distributions
from gscore import peakgroups
from gscore.parsers.osw import (
    osw,
    queries
)

def plot_distributions(
        target_distribution,
        null_distribution,
        all_targets_distribution,
        fig_path
):

    fig, ax = plt.subplots()
    sns.lineplot(
        x=target_distribution.x_axis,
        y=target_distribution.values,
        ax=ax,
        label='Targets'
    )
    sns.lineplot(
        x=null_distribution.x_axis,
        y=null_distribution.values,
        ax=ax,
        label='False Targets'
    )
    sns.lineplot(
        x=all_targets_distribution.x_axis,
        y=all_targets_distribution.values,
        ax=ax,
        label='All Targets'
    )
    plt.savefig(fig_path)


def main(args):
    print(args.input_files)

    if args.use_decoys:

        pass

    else:

        osw_query = queries.SelectPeakGroups.FETCH_SCORED_DATA_DECOY_FREE


    if args.scoring_level == 'peptide':

        print('Parsing files to graph')

        full_graph = osw.fetch_peptide_level_global_graph(
            args.input_files,
            osw_query,
            args.score_column
        )

        peakgroups.get_median_peptide_vote_percentage(full_graph)

    elif args.scoring_level == 'protein':

        pass

    true_targets = full_graph.query_nodes(
        color='peptide',
        rank=1,
        query="probability > 0.5"
    )

    false_targets = full_graph.query_nodes(
        color='peptide',
        rank=1,
        query="probability < 0.5"
    )

    all_targets = full_graph.query_nodes(
        color='peptide',
        rank=1
    )

    target_scores = peakgroups.get_score_array(
        graph=full_graph,
        node_list=true_targets,
        score_column=args.score_column
    )

    false_target_scores = peakgroups.get_score_array(
        graph=full_graph,
        node_list=false_targets,
        score_column=args.score_column
    )

    all_scores = peakgroups.get_score_array(
        graph=full_graph,
        node_list=all_targets,
        score_column=args.score_column
    )

    target_distribution = distributions.LabelDistribution(
        data=target_scores.reshape(-1, 1)
    )

    false_target_distribution = distributions.LabelDistribution(
        data=false_target_scores.reshape(-1, 1)
    )

    all_targets_distribution = distributions.LabelDistribution(
        data=all_scores.reshape(-1, 1)
    )

    source_path = Path(args.input_files[0]).parent

    plot_distributions(
        target_distribution,
        false_target_distribution,
        all_targets_distribution,
        fig_path=f"{source_path}/{args.scoring_level}.score_distribution.png"
    )

    score_distribution = distributions.ScoreDistribution(
        null_distribution=false_target_distribution,
        target_distribution=target_distribution,
    )

    with open(args.model_output, 'wb') as pkl:
        pickle.dump(score_distribution, pkl)
