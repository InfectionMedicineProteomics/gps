import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score

from gscore.utils.connection import Connection
from gscore.parsers.osw import (
    osw,
    queries
)
from gscore import peakgroups, denoiser, distributions


def plot_distributions(target_distribution, null_distribution, fig_path):

    fig, ax = plt.subplots()
    sns.lineplot(x=target_distribution.x_axis, y=target_distribution.values, ax=ax, label='Targets')
    sns.lineplot(x=null_distribution.x_axis, y=null_distribution.values, ax=ax, label='False Targets')
    plt.savefig(fig_path)

def prepare_qvalue_add_records(graph):

    record_updates = list()

    for peakgroup in graph.iter(color='peakgroup'):

        record = {
            'feature_id': peakgroup.data.key,
            'vote_percentage': peakgroup.data.scores['vote_percentage'],
            'probability': peakgroup.data.scores['probability'],
            'logit_probability': peakgroup.data.scores['logit_probability'],
            'd_score': peakgroup.data.scores['d_score'],
            'weighted_d_score': peakgroup.data.scores['weighted_d_score'],
            'q_value': peakgroup.data.scores['q_value']
        }

        record_updates.append(record)

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

        training_peptides = peakgroups.get_training_peptides(
            peptide_folds=peptide_folds,
            fold_num=fold_num
        )

        denoizer, scaler = denoiser.get_denoizer(
            peakgroup_graph,
            training_peptides,
            n_estimators=args.num_classifiers,
            n_jobs=args.threads
        )

        testing_scores, testing_labels, testing_indices = peakgroups.preprocess_data_to_score(
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

        peakgroups.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            vote_percentages,
            "vote_percentage"
        )

        peakgroups.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            probabilities,
            "probability"
        )

        peakgroups.update_peakgroup_scores(
            peakgroup_graph,
            testing_indices,
            logit_probabilities,
            "logit_probability"
        )

        val_scores, val_labels, _ = peakgroups.preprocess_training_data(
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

        with open(args.scaler_path, 'rb') as pkl:
            scaler_pipeline = pickle.load(pkl)

        scoring_model = tf.keras.models.load_model(
            args.model_path
        )

        all_peakgroups = peakgroup_graph.get_nodes(color='peakgroup')
        all_peptides = peakgroup_graph.get_nodes(color='peptide')

        for node in peakgroup_graph.iter(keys=all_peakgroups):

            probability = node.data.scores['probability']

            node.data.add_sub_score_column(
                key='probability',
                value=probability
            )

        scores, labels, indices = peakgroups.preprocess_data_to_score(
            peakgroup_graph,
            list(all_peptides),
            return_all=True
        )

        scores = scaler_pipeline.transform(scores)

        d_scores = scoring_model.predict(
            scores
        ).ravel()

        vote_percentages = list()

        for index in indices:
            vote_percentage = peakgroup_graph[index].data.scores['vote_percentage']

            vote_percentages.append(
                vote_percentage
            )
        vote_percentages = np.array(vote_percentages)

        weighted_d_scores = np.exp(
            vote_percentages
        ) * d_scores

        for d_score, weighted_d_score, index in zip(d_scores, weighted_d_scores, indices):

            peakgroup_graph[index].data.scores['d_score'] = d_score

            peakgroup_graph[index].data.scores['weighted_d_score'] = weighted_d_score

            peptide_id = list(peakgroup_graph[index]._edges.keys())[0]

            peakgroup_graph.update_edge_weight(
                node_from=peptide_id,
                node_to=index,
                weight=weighted_d_score,
                directed=False
            )

        true_targets = peakgroup_graph.query_nodes(
            color='peptide',
            rank=1,
            query="vote_percentage == 1.0"
        )

        false_targets = peakgroup_graph.query_nodes(
            color='peptide',
            rank=1,
            query="vote_percentage < 1.0"
        )

        second_ranked = peakgroup_graph.query_nodes(
            color='peptide',
            rank=2,
        )

        target_scores = peakgroups.get_score_array(
            graph=peakgroup_graph,
            node_list=true_targets,
            score_column='weighted_d_score'
        )

        false_target_scores = peakgroups.get_score_array(
            graph=peakgroup_graph,
            node_list=false_targets,
            score_column='weighted_d_score'
        )

        second_target_scores = peakgroups.get_score_array(
            graph=peakgroup_graph,
            node_list=second_ranked,
            score_column='weighted_d_score'
        )

        target_distribution = distributions.LabelDistribution(
            data=target_scores.reshape(-1, 1)
        )

        false_target_distribution = distributions.LabelDistribution(
            data=false_target_scores.reshape(-1, 1)
        )

        score_distribution = distributions.ScoreDistribution(
            null_distribution=false_target_distribution,
            target_distribution=target_distribution,
        )

        plot_distributions(
            target_distribution,
            false_target_distribution,
            f"{args.input}.score_distribution.png"
        )

        all_peak_groups = peakgroup_graph.query_nodes(
            color='peptide',
            return_all=True
        )

        for node in peakgroup_graph.iter(keys=all_peak_groups):

            score = node.data.scores['weighted_d_score']

            node.data.scores['q_value'] = score_distribution.calc_q_value(score)


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
