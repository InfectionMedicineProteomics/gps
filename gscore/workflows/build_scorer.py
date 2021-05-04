import pickle

import pandas as pd
import numpy as np


from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score


from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler
)

from gscore.utils.connection import Connection

from gscore.parsers.osw import peakgroups

from gscore.parsers.osw.queries import (
    SelectPeakGroups
)

from gscore.models.denoiser import (
    BaggedDenoiser
)

import tensorflow as tf

from gscore.models import scorer
from gscore.models import preprocess


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

    input_files = [input_file.name for input_file in args.input_files]

    all_training_records = list()

    for input_path in input_files:

        logger.info(f"Denoising {input_path}")

        peak_group_records = list()

        with Connection(input_path) as conn:

            for record in conn.iterate_records(
                SelectPeakGroups.FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE
            ):
                peak_group_records.append(record)

        print("Processing peakgroups")

        peak_groups = peakgroups.preprocess_data(
            pd.DataFrame(peak_group_records)
        )

        split_peak_groups = dict(
            tuple(
                peak_groups.groupby('transition_group_id')
            )
        )

        peak_groups = peakgroups.PeakGroupList(
            split_peak_groups
        )

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

        shuffled_peak_groups = noisey_target_labels.sample(frac=1)

        split_data = np.array_split(
            shuffled_peak_groups,
            args.num_folds
        )

        all_peak_groups = peak_groups.select_peak_group(
            return_all=True
        )

        scored_data = list()

        print("Denoising target labels")

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

            n_samples = int(len(swath_training_prepared) * 0.5) # Change this later based on sample size

            denoizer = BaggedDenoiser(
                max_samples=n_samples,
                n_estimators=args.num_classifiers,
                threads=args.threads
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

        split_peak_groups = dict(
            tuple(
                scored_data.groupby('transition_group_id')
            )
        )

        peak_groups = peakgroups.PeakGroupList(split_peak_groups)

        highest_ranking = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['logit_probability'],
            ascending=False
        )

        low_ranking = list()

        for rank in range(2, 3):
            lower_ranking = peak_groups.select_peak_group(
                rank=rank,
                rerank_keys=['logit_probability'],
                ascending=False
            )

            low_ranking.append(lower_ranking)

        low_ranking = pd.concat(
            low_ranking,
            ignore_index=True
        )

        targets = highest_ranking[
            (highest_ranking['probability'] >= 0.9)
        ].copy()

        false_targets = low_ranking[
            (low_ranking['probability'] < 0.5)
        ].copy()
        
        false_targets['target'] = 0.0

        denoised_labels = pd.concat(
            [
                targets,
                false_targets
            ]
        )
        
        print(denoised_labels.target.value_counts())
        
        all_training_records.append(
            denoised_labels
        )

    all_training_records = pd.concat(all_training_records)

    print(all_training_records.target.value_counts())

    training_split, test_split = train_test_split(
        all_training_records
    )

    scoring_columns = [
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
        'var_yseries_score',
        # 'vote_percentage',
        # 'probabilities',
        'logit_probability'
        # 'ensemble_score'
    ]

    learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        0.01,
        1000,
        0.001
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    training_preprocessed, scaling_pipeline = preprocess.preprocess_data(
        pipeline=preprocess.STANDARD_SCALAR_PIPELINE,
        data=training_split.copy(),
        columns=scoring_columns,
        train=True,
        return_scaler=True
    )

    dense_model = scorer.TargetScoringModel(
        input_dim=training_preprocessed[scoring_columns].shape[1:]
    )

    dense_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[
            'accuracy'
        ]
    )

    dense_history = dense_model.fit(
        training_preprocessed[scoring_columns],
        training_preprocessed['target'],
        epochs=100,
        validation_split=0.10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True
            )
        ],
        batch_size=100
    )

    testing_processed = preprocess.preprocess_data(
        pipeline=scaling_pipeline,
        data=test_split.copy(),
        columns=scoring_columns,
    )

    dense_model.evaluate(testing_processed[scoring_columns], testing_processed['target'])

    print('saving trained model and scaler')

    dense_model.save(
        f'{args.output_directory}/{args.model_name}.h5'
    )

    scaler_file_path = f'{args.output_directory}/{args.model_name}.scaler.pkl'

    with open(scaler_file_path, 'wb') as pkl:
        pickle.dump(scaling_pipeline, pkl)
