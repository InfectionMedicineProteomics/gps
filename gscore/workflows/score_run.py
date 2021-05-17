import random

from gscore.utils.connection import Connection

import pandas as pd
import numpy as np
import seaborn as sns

import gscore.datastructures as graph_funcs

from gscore import peakgroups

from gscore.parsers.osw import (
    osw,
    queries
)

# Need to rename the preprocess function
from gscore.peakgroups import PeakGroupList

from gscore.models import denoiser

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score


from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)

import pickle


from gscore.models import preprocess
from gscore.models.distributions import ScoreDistribution

import tensorflow as tf


def prepare_add_denoise_records(records):

    record_updates = records[
        [
            'feature_id',
            'vote_percentage',
            'probability',
            'logit_probability'
        ]
    ]

    feature_ids = list(record_updates['feature_id'])

    votes = list(record_updates['vote_percentage'])

    probabilities = list(record_updates['probability'])

    logit_probabilities = list(record_updates['logit_probability'])

    record_updates = list()

    for feature_id, vote, probability, logit_probability in zip(
            feature_ids, votes, probabilities, logit_probabilities
    ):
        record_updates.append(
            {
                'feature_id': feature_id,
                'vote_percentage': vote,
                'probability': probability,
                'logit_probability': logit_probability
            }
        )

    return record_updates


def prepare_denoise_record_additions(graph):
    record_updates = list()

    for peakgroup in graph.iter(color='peakgroup'):
        record = {
            'feature_id': peakgroup.data.key,
            'vote_percentage': peakgroup.data.scores['vote_percentage'],
            'probability': peakgroup.data.scores['probability'],
            'logit_probability': peakgroup.data.scores['logit_probability'],
        }

        record_updates.append(record)

    return record_updates


def main(args, logger):

    peakgroup_graph = osw.fetch_peakgroup_graph(
        osw_path=args.input,
        osw_query=queries.SelectPeakGroups.FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE
    )

    print(f"Denoising {args.input}")

    print("Processing peakgroups")

    peptide_ids = list(peakgroup_graph.get_nodes(color='peptide'))

    random.seed(42)
    random.shuffle(peptide_ids)

    peptide_folds = np.array_split(
        peptide_ids,
        args.num_folds
    )

    for fold_num, peptide_fold in enumerate(peptide_folds):

        print(f"Processing fold {fold_num + 1}")

        training_peptides = graph_funcs.get_training_peptides(
            peptide_folds=peptide_folds,
            fold_num=fold_num
        )

        denoizer, scaler = denoiser.get_denoizer(
            peakgroup_graph,
            training_peptides,
            n_estimators=args.num_classifiers,
            n_jobs=args.threads
        )

        testing_scores, testing_labels, testing_indices = graph_funcs.preprocess_data_to_score(
            peakgroup_graph,
            list(peptide_fold),
            return_all=True

        )

        testing_scores = scaler.transform(
            testing_scores
        )

        class_index = np.where(
            denoizer.classes_ == 1.0
        )[0][0]

        print("Scoring data")

        vote_percentages = denoizer.vote(
            testing_scores,
            threshold=0.5
        )

        probabilities = denoizer.predict_proba(
            testing_scores
        )[:, class_index]

        logit_probabilities = np.log(
            (
                    probabilities / (1 - probabilities)
            )
        )

        print("Updating peakgroups")

        graph_funcs.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            vote_percentages,
            "vote_percentage"
        )

        graph_funcs.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            probabilities,
            "probability"
        )

        graph_funcs.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            logit_probabilities,
            "logit_probability"
        )

        val_scores, val_labels, _ = graph_funcs.preprocess_training_data(
            peakgroup_graph,
            list(peptide_fold),
        )

        val_scores = scaler.transform(
            val_scores
        )

        fold_precision = precision_score(
            y_pred=denoizer.predict(val_scores),
            y_true=val_labels.ravel()
        )

        fold_recall = recall_score(
            y_pred=denoizer.predict(val_scores),
            y_true=val_labels.ravel()
        )

        print(
            f"Fold {fold_num + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
        )

    if args.denoise_only:

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

    else:

        pass

        # with open(args.scaler_path, 'rb') as pkl:
        #     scaler_pipeline = pickle.load(pkl)
        #
        # scoring_model = tf.keras.models.load_model(
        #     args.model_path
        # )
        #
        # scoring_columns = score_columns + ['logit_probability']
        #
        # all_peak_groups = preprocess.preprocess_data(
        #     pipeline=scaler_pipeline,
        #     data=scored_data.copy(),
        #     columns=scoring_columns
        # )
        #
        # print(f'Scoring {args.input}')
        #
        # all_peak_groups['d_score'] = scoring_model.predict(
        #     all_peak_groups[scoring_columns]
        # ).ravel()
        #
        # all_peak_groups['weighted_d_score'] = np.exp(
        #     all_peak_groups['vote_percentage']
        # ) * all_peak_groups['d_score']
        #
        # split_peak_groups = dict(
        #     tuple(
        #         all_peak_groups.groupby('transition_group_id')
        #     )
        # )
        #
        # peak_groups = PeakGroupList(split_peak_groups)
        #
        # print('Modelling run and calculating q-values')
        #
        # #TODO: Need to have the peak selection in another function
        #
        # highest_ranking = peak_groups.select_peak_group(
        #     rank=1,
        #     rerank_keys=['weighted_d_score'],
        #     ascending=False
        # )
        #
        # low_ranking = list()
        #
        # for rank in range(2, 3):
        #     lower_ranking = peak_groups.select_peak_group(
        #         rank=rank,
        #         rerank_keys=['weighted_d_score'],
        #         ascending=False
        #     )
        #
        #     low_ranking.append(lower_ranking)
        #
        # low_ranking = pd.concat(
        #     low_ranking,
        #     ignore_index=True
        # )
        #
        # low_ranking['target'] = 0.0
        #
        # low_ranking = low_ranking[
        #     low_ranking['vote_percentage'] < 1.0
        # ].copy()
        #
        # model_distribution = pd.concat(
        #     [
        #         highest_ranking,
        #         low_ranking
        #     ]
        # )
        #
        # #TODO: May should save the actual learned distribution from below
        # # instead of the nice looking one
        # run_distributions_plot = sns.displot(
        #     model_distribution,
        #     x='weighted_d_score',
        #     hue='target',
        #     element='step',
        #     kde=True
        # )
        #
        # run_distributions_plot.savefig(f"{args.input}.score_distribution.png")
        #
        # print('building score distributions')
        #
        # score_distribution = ScoreDistribution(
        #     data=model_distribution,
        #     distribution_type='weighted_d_score'
        # )
        #
        # all_peak_groups = peak_groups.select_peak_group(
        #     return_all=True
        # )
        #
        # all_peak_groups['q_value'] = all_peak_groups['weighted_d_score'].apply(
        #     score_distribution.calc_q_value
        # )
        #
        # print(f'Updating {args.input}')
        #
        # record_updates = prepare_add_records(
        #     all_peak_groups
        # )
        #
        # with Connection(args.input) as conn:
        #     conn.drop_table(
        #         'ghost_score_table'
        #     )
        #
        #     conn.create_table(
        #         CreateTable.CREATE_GHOSTSCORE_TABLE
        #     )
        #
        #     conn.add_records(
        #         table_name='ghost_score_table',
        #         records=record_updates
        #     )
