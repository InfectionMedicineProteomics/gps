import pickle

import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from gscore.utils.connection import Connection

from gscore import peakgroups

from gscore.parsers.osw.queries import (
    SelectPeakGroups
)

from gscore.parsers.osw import osw


import tensorflow as tf

from gscore.models import scorer
from gscore.models import preprocess


def main(args, logger):

    input_files = [input_file.name for input_file in args.input_files]

    all_training_records = list()

    for input_path in input_files:

        logger.info(f"Processing {input_path}")

        peak_group_records = list()

        with Connection(input_path) as conn:

            for record in conn.iterate_records(
                SelectPeakGroups.FETCH_VOTED_DATA_DECOY_FREE
            ):
                peak_group_records.append(record)

        peak_groups = osw.preprocess_data(
            pd.DataFrame(peak_group_records)
        )

        split_peak_groups = dict(
            tuple(
                peak_groups.groupby('transition_group_id')
            )
        )

        print("Processing peak groups")

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

        print("Spliting targets/false targets")

        targets = highest_ranking[
            (highest_ranking['vote_percentage'] == 1.0)
        ].copy()

        # targets_below_50 = highest_ranking[
        #     (highest_ranking['probability'] < 0.2)
        # ].copy()
        #
        # targets_below_50['target'] = 0.0
        #
        # num_targets_below_50 = len(targets_below_50)
        #
        # print(f"Targets below 50 {num_targets_below_50}")

        false_targets = low_ranking[
            (low_ranking['vote_percentage'] < 0.5)
        ].copy()

        false_targets['target'] = 0.0

        denoised_labels = pd.concat(
            [
                targets,
                false_targets,
                #targets_below_50
            ]
        )

        print(denoised_labels.target.value_counts())

        all_training_records.append(
            denoised_labels
        )

    all_training_records = pd.concat(all_training_records)

    print(all_training_records.target.value_counts())

    targets = all_training_records[
        all_training_records['target'] == 1.0
    ].copy()

    false_targets = all_training_records[
        all_training_records['target'] == 0.0
    ].copy()

    num_false_targets = len(false_targets)
    num_targets = len(targets)

    if num_false_targets < num_targets:
        targets = resample(
            targets,
            replace=False,
            n_samples=num_false_targets,
            random_state=0
        )
    elif num_false_targets > num_targets:
        false_targets = resample(
            false_targets,
            replace=False,
            n_samples=num_targets,
            random_state=0
        )

    all_training_records = pd.concat(
        [
            targets,
            false_targets
        ]
    )

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

    # learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    #     0.01,
    #     1000,
    #     0.001
    # )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

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
        epochs=500,
        validation_split=0.10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True
            )
        ],
        batch_size=32,
        shuffle=True
    )

    testing_processed = preprocess.preprocess_data(
        pipeline=scaling_pipeline,
        data=test_split.copy(),
        columns=scoring_columns,
    )

    dense_model.evaluate(testing_processed[scoring_columns], testing_processed['target'])

    print('saving trained model and scaler')

    dense_model.save(
        f'{args.output_directory}/{args.model_name}',
        save_format='tf'
    )

    scaler_file_path = f'{args.output_directory}/{args.model_name}.scaler.pkl'

    with open(scaler_file_path, 'wb') as pkl:
        pickle.dump(scaling_pipeline, pkl)
