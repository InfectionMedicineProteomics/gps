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
from gscore.parsers.osw import (
    osw,
    queries
)

def plot_distributions(
        all_scores,
        target_distribution,
        null_distribution,
        all_targets_distribution,
        fig_path
):

    fig, ax = plt.subplots()
    plt.hist(all_scores, bins='auto', density=True)
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

    print(f'Building {args.scoring_level} scoring model')

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

        peakgroups.calc_score_grouped_by_level(
            full_graph,
            function=np.median,
            level='peptide',
            score_column='vote_percentage',
            new_column_name='median_vote_percentage'
        )

        true_targets = full_graph.query_nodes(
            color='peptide',
            rank=1,
            query=f"probability >= {args.true_target_cutoff}"
        )

        false_targets = full_graph.query_nodes(
            color='peptide',
            rank=1,
            query=f"probability < {args.false_target_cutoff}"
        )

        all_targets = full_graph.query_nodes(
            color='peptide',
            rank=1
        )

    elif args.scoring_level == 'protein':

        print('Parsing files to graph')

        full_graph = osw.fetch_protein_level_global_graph(
            args.input_files,
            osw_query,
            args.score_column
        )

        peakgroups.calc_score_grouped_by_level(
            full_graph,
            function=np.median,
            level='protein',
            score_column='vote_percentage',
            new_column_name='median_vote_percentage'
        )

        true_targets = full_graph.query_nodes(
            color='protein',
            rank=1,
            query=f"probability >= {args.true_target_cutoff}"
        )

        false_targets = full_graph.query_nodes(
            color='protein',
            rank=1,
            query=f"probability < {args.false_target_cutoff}"
        )

        all_targets = full_graph.query_nodes(
            color='protein',
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

    score_mixture_model = GeneralMixtureModel(
        [
            NormalDistribution(
                false_target_scores.mean(),
                false_target_scores.std()
            ),
            NormalDistribution(
                target_scores.mean(),
                target_scores.std()
            )
        ]
    )

    score_mixture_model.fit(all_scores)


    target_distribution = distributions.LabelDistribution(
        data=all_scores,
        axis_span=(
            all_scores.min() - 10,
            all_scores.max() + 10,
        ),
        model=score_mixture_model.distributions[1]
    )

    false_target_distribution = distributions.LabelDistribution(
        data=all_scores,
        axis_span=(
            all_scores.min() - 10,
            all_scores.max() + 10,
        ),
        model = score_mixture_model.distributions[0]
    )

    all_targets_distribution = distributions.LabelDistribution(
        data=all_scores,
        axis_span=(
            all_scores.min() - 10,
            all_scores.max() + 10,
        ),
        model=score_mixture_model
    )

    source_path = Path(args.input_files[0]).parent

    plot_distributions(
        all_scores,
        target_distribution,
        false_target_distribution,
        all_targets_distribution,
        fig_path=f"{source_path}/{args.scoring_level}.score_distribution.png"
    )

    score_distribution = distributions.ScoreDistribution(
        null_distribution=false_target_distribution,
        target_distribution=all_targets_distribution,
    )

    if args.model_output:

        with open(args.model_output, 'wb') as pkl:
            pickle.dump(score_distribution, pkl)

    return score_distribution
