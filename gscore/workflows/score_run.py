import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pomegranate import (
    GeneralMixtureModel,
    NormalDistribution
)

from sklearn.metrics import precision_score, recall_score

from gscore.denoiser import Scaler, BaggedDenoiser
from gscore.utils.connection import Connection
from gscore.parsers import osw, queries
from gscore import peakgroups, denoiser, distributions
from gscore.distributions import ScoreDistribution

import networkx as nx
from typing import List, Dict

def plot_distributions(
        target_distribution,
        null_distribution,
        x_axis,
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

def prepare_qvalue_add_records(graph):

    record_updates = list()

    for peakgroup_key in graph.get_nodes(color='peakgroup'):

        peakgroup = graph[peakgroup_key]

        record = {
            'feature_id': peakgroup.key,
            'probability': peakgroup.scores['probability'],
            'd_score': peakgroup.scores['d_score'],
            'q_value': peakgroup.scores['q_value']
        }

        record_updates.append(record)

    return record_updates


def main(args, logger=None):

    if args.apply_model:

        print(f"Loading peakgroups from {args.input}")

        peakgroup_graph, _ = osw.fetch_peakgroup_graph(
            osw_path=args.input,
            query=queries.SelectPeakGroups.FETCH_ALL_UNSCORED_DATA
        )

        print("Denoising.")

        denoiser.denoise(
            peakgroup_graph,
            args.num_folds,
            args.num_classifiers,
            args.threads,
            args.vote_threshold
        )

        print(f"Applying model to {args.input}")

        with open(args.scaler_path, 'rb') as pkl:
            scaler_pipeline = pickle.load(pkl)

        scoring_model = tf.keras.models.load_model(
            args.model_path
        )

        peakgroups = list()

        for node_key in peakgroup_graph.get_nodes("peakgroup"):

            node = peakgroup_graph[node_key]

            peakgroups.append(node)

        print("Reformatting data")
        scores, labels, indices = reformat_data(
            peakgroups,
            include_score_columns=True
        )

        print("Transforming data")
        scores = scaler_pipeline.transform(scores)


        print("Calculating probabilities")
        probabilities = scoring_model.predict(
            scores
        ).ravel()

        d_scores = np.log(
            (probabilities / (1 - probabilities))
        )

        for idx, d_score, probability in zip(indices, d_scores, probabilities):

            peakgroup_graph[idx].scores['d_score'] = d_score

            peakgroup_graph[idx].scores['probability'] = probability

        if args.peakgroup_scoring_model_path:

            print("Loading peakgroup scoring model.")

            with open(args.peakgroup_scoring_model_path, 'rb') as pkl:

                score_distribution = pickle.load(pkl)

        else:

            target_peakgroups = peakgroup_graph.get_ranked_peakgroups(
                rank=1,
                target=1
            )

            decoy_peakgroups = peakgroup_graph.get_ranked_peakgroups(
                rank=1,
                target=0
            )

            target_scores = [
                target_peakgroup.scores['d_score'] for target_peakgroup in target_peakgroups
            ]

            decoy_scores = [
                decoy_peakgroup.scores['d_score'] for decoy_peakgroup in decoy_peakgroups
            ]

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

            plot_distributions(
                target_distribution.values(x_plot),
                decoy_distribution.values(x_plot),
                x_plot.ravel(),
                f"{args.input}.score_distribution.png"
            )

        print("Calculating q-values.")

        for node_key in peakgroup_graph.get_nodes("peakgroup"):

            node = peakgroup_graph[node_key]

            score = node.scores['d_score']

            node.scores['q_value'] = score_distribution.calc_q_value(score)

        print(f'Updating {args.input}')

        record_updates = prepare_qvalue_add_records(
            peakgroup_graph
        )

        with Connection(args.input) as conn:

            conn.drop_table(
                'ghost_score_table'
            )

            conn.create_table(
                queries.CreateTable.CREATE_GHOSTSCORE_TABLE
            )

            conn.add_records(
                table_name='ghost_score_table',
                records=record_updates
            )

    else:

        print(f"Denoising {args.input}")

        print("Processing peakgroups")

    if args.denoise_only:
        peakgroup_graph, _ = osw.fetch_peakgroup_graph(
            osw_path=args.input,
        )

        denoiser.denoise(
            peakgroup_graph,
            args.num_folds,
            args.num_classifiers,
            args.threads,
            args.vote_threshold
        )

        record_updates = prepare_denoise_record_additions(
            peakgroup_graph
        )



        with Connection(args.input) as conn:
            conn.drop_table(
                'ghost_score_table'
            )

            conn.create_table(
                queries.CreateTable.CREATE_GHOSTSCORE_TABLE
            )

            conn.add_records(
                table_name='ghost_score_table',
                records=record_updates
            )
