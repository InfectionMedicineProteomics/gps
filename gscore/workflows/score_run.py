from gscore.utils.connection import Connection

import pandas as pd
import numpy as np
import seaborn as sns

from gscore import peakgroups

from gscore.parsers.osw import osw
from gscore.parsers.osw.queries import (
    SelectPeakGroups,
    CreateTable
)

# Need to rename the preprocess function
from gscore.peakgroups import PeakGroupList

from gscore.models.denoiser import (
    BaggedDenoiser
)

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

def prepare_add_records(records):
    record_updates = records[
        [
            'feature_id',
            'vote_percentage',
            'probability',
            'logit_probability',
            'd_score',
            'weighted_d_score',
            'q_value'
        ]
    ]

    feature_ids = list(record_updates['feature_id'])

    votes = list(record_updates['vote_percentage'])

    d_scores = list(record_updates['d_score'])

    weighted_d_scores = list(record_updates['weighted_d_score'])

    probabilities = list(record_updates['probability'])

    logit_probabilities = list(record_updates['logit_probability'])

    q_values = list(record_updates['q_value'])

    record_updates = list()

    for feature_id, vote, d_score, weighted_d_score, probability, logit_probability, q_value in zip(
            feature_ids, votes, d_scores, weighted_d_scores, probabilities, logit_probabilities, q_values
    ):
        record_updates.append(
            {'feature_id': feature_id,
             'vote_percentage': vote,
             'probability': probability,
             'logit_probability': logit_probability,
             'd_score': d_score,
             'weighted_d_score': weighted_d_score,
             'm_score': q_value}
        )

    return record_updates


def add_vote_records(records, osw_path):

    with Connection(osw_path) as conn:

        conn.add_records(
            table_name='ghost_score_table', 
            records=records
        )

def main(args, logger):

    score_columns = [
        'var_massdev_score_ms1',
        'var_isotope_correlation_score_ms1',
        'var_isotope_overlap_score_ms1',
        'var_xcorr_coelution_contrast_ms1',
        'var_xcorr_coelution_combined_ms1',
        'var_xcorr_shape_contrast_ms1',
        'var_xcorr_shape_combined_ms1',
        'var_bseries_score',
        'var_dotprod_score',
        'var_intensity_score',
        'var_isotope_correlation_score',
        'var_isotope_overlap_score',
        'var_library_corr',
        'var_library_dotprod',
        'var_library_manhattan',
        'var_library_rmsd',
        'var_library_rootmeansquare',
        'var_library_sangle',
        'var_log_sn_score',
        'var_manhattan_score',
        'var_massdev_score',
        'var_massdev_score_weighted',
        'var_norm_rt_score',
        'var_xcorr_coelution',
        'var_xcorr_coelution_weighted',
        'var_xcorr_shape',
        'var_xcorr_shape_weighted',
        'var_yseries_score'
    ]

    print(f"Denoising {args.input}")
    print("Processing peakgroups")
    peak_group_records = list()

    #TODO: Refactor this into a separate function
    with Connection(args.input) as conn:
        for record in conn.iterate_records(SelectPeakGroups.FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE):
            peak_group_records.append(record)

    peak_groups = osw.preprocess_data(
        pd.DataFrame(peak_group_records)
    )

    split_peak_groups = dict(
        tuple(
            peak_groups.groupby('transition_group_id')
        )
    )

    peak_groups = peakgroups.PeakGroupList(split_peak_groups)

    #TODO: Reimplement the peakgroup datastructure to not use
    highest_ranking = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['var_xcorr_shape_weighted'],
        ascending=False
    )

    low_ranking = list()

    for rank in range(2, 3):
        lower_ranking = peak_groups.select_peak_group(
            rank=rank,
            rerank_keys=['var_xcorr_shape_weighted'],
            ascending=False
        )

        low_ranking.append(lower_ranking)

    low_ranking = pd.concat(
        low_ranking,
        ignore_index=True
    )

    low_ranking['target'] = 0.0

    noisey_target_labels = pd.concat(
        [
            highest_ranking,
            low_ranking
        ]
    )

    print(noisey_target_labels.target.value_counts())

    #TODO: Break shuffling of peakgroups and splitting of data into
    # different functions
    shuffled_peak_groups = noisey_target_labels.sample(frac=1)

    split_data = np.array_split(
        shuffled_peak_groups,
        args.num_folds
    )

    split_data = split_data[::-1]

    all_peak_groups = peak_groups.select_peak_group(
        return_all=True
    )

    scored_data = list()

    print("Denoising target labels")

    #TODO: Refactor denoising to a specific function
    for idx, fold_data in enumerate(split_data):
        training_data = pd.concat(
            [df for i, df in enumerate(split_data) if i != idx]
        )

        full_pipeline = Pipeline([
            ('standard_scaler', RobustScaler()),
            ('min_max_scaler', MinMaxScaler())
        ])

        swath_training_prepared = training_data.copy()

        swath_training_prepared[score_columns] = full_pipeline.fit_transform(
            swath_training_prepared[score_columns]
        )

        print(
            f"Number of labels: \n",
            swath_training_prepared.target.value_counts()
        )

        n_samples = int(len(swath_training_prepared) * 1.0)  # Change this later based on sample size

        denoizer = BaggedDenoiser(
            max_samples=n_samples,
            n_estimators=args.num_classifiers,
            threads=args.threads,
            random_state=idx
        )

        denoizer.fit(
            swath_training_prepared[score_columns],
            swath_training_prepared['target']
        )

        group_ids = list(fold_data['transition_group_id'])

        left_out_peak_groups = all_peak_groups.loc[
            all_peak_groups['transition_group_id'].isin(group_ids)
        ].copy()

        left_out_peak_groups_transformed = left_out_peak_groups.copy()

        left_out_peak_groups_transformed[score_columns] = full_pipeline.transform(
            left_out_peak_groups[score_columns]
        )

        left_out_peak_groups['vote_percentage'] = denoizer.vote(
            left_out_peak_groups_transformed[score_columns]
        )

        class_index = np.where(
            denoizer.classes_ == 1.0
        )[0][0]

        left_out_peak_groups['probability'] = denoizer.predict_proba(
            left_out_peak_groups_transformed[score_columns]
        )[:, class_index]

        left_out_peak_groups['logit_probability'] = np.log(
            (
                    left_out_peak_groups['probability'] / (1 - left_out_peak_groups['probability'])
            )
        )

        fold_data[score_columns] = full_pipeline.transform(
            fold_data[score_columns]
        )

        fold_precision = precision_score(
            denoizer.predict(
                fold_data[score_columns]
            ),
            fold_data['target']
        )

        fold_recall = recall_score(
            denoizer.predict(
                fold_data[score_columns]
            ),
            fold_data['target']
        )

        print(
            f"Fold {idx + 1}: Precision = {fold_precision}, Recall = {fold_recall}"
        )

        untransformed_peak_groups = all_peak_groups.loc[
            all_peak_groups['transition_group_id'].isin(group_ids)
        ].copy()

        untransformed_peak_groups['vote_percentage'] = left_out_peak_groups['vote_percentage']
        untransformed_peak_groups['probability'] = left_out_peak_groups['probability']
        untransformed_peak_groups['logit_probability'] = left_out_peak_groups['logit_probability']

        scored_data.append(untransformed_peak_groups)

    scored_data = pd.concat(
        scored_data,
        ignore_index=True
    )

    if args.denoise_only:

        record_updates = prepare_add_denoise_records(
            scored_data
        )

        with Connection(args.input) as conn:
            conn.drop_table(
                'ghost_score_table'
            )

            conn.create_table(
                CreateTable.CREATE_GHOSTSCORE_TABLE
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

        scoring_columns = score_columns + ['logit_probability']

        all_peak_groups = preprocess.preprocess_data(
            pipeline=scaler_pipeline,
            data=scored_data.copy(),
            columns=scoring_columns
        )

        print(f'Scoring {args.input}')

        all_peak_groups['d_score'] = scoring_model.predict(
            all_peak_groups[scoring_columns]
        ).ravel()

        all_peak_groups['weighted_d_score'] = np.exp(
            all_peak_groups['vote_percentage']
        ) * all_peak_groups['d_score']

        split_peak_groups = dict(
            tuple(
                all_peak_groups.groupby('transition_group_id')
            )
        )

        peak_groups = PeakGroupList(split_peak_groups)

        print('Modelling run and calculating q-values')

        #TODO: Need to have the peak selection in another function

        highest_ranking = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['weighted_d_score'],
            ascending=False
        )

        low_ranking = list()

        for rank in range(2, 3):
            lower_ranking = peak_groups.select_peak_group(
                rank=rank,
                rerank_keys=['weighted_d_score'],
                ascending=False
            )

            low_ranking.append(lower_ranking)

        low_ranking = pd.concat(
            low_ranking,
            ignore_index=True
        )

        low_ranking['target'] = 0.0

        low_ranking = low_ranking[
            low_ranking['vote_percentage'] < 1.0
        ].copy()

        model_distribution = pd.concat(
            [
                highest_ranking,
                low_ranking
            ]
        )

        #TODO: May should save the actual learned distribution from below
        # instead of the nice looking one
        run_distributions_plot = sns.displot(
            model_distribution,
            x='weighted_d_score',
            hue='target',
            element='step',
            kde=True
        )

        run_distributions_plot.savefig(f"{args.input}.score_distribution.png")

        print('building score distributions')

        score_distribution = ScoreDistribution(
            data=model_distribution,
            distribution_type='weighted_d_score'
        )

        all_peak_groups = peak_groups.select_peak_group(
            return_all=True
        )

        all_peak_groups['q_value'] = all_peak_groups['weighted_d_score'].apply(
            score_distribution.calc_q_value
        )

        print(f'Updating {args.input}')

        record_updates = prepare_add_records(
            all_peak_groups
        )

        with Connection(args.input) as conn:
            conn.drop_table(
                'ghost_score_table'
            )

            conn.create_table(
                CreateTable.CREATE_GHOSTSCORE_TABLE
            )

            conn.add_records(
                table_name='ghost_score_table',
                records=record_updates
            )
