import pickle

import pandas as pd
import tensorflow as tf

from tensorflow import keras


from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    SelectPeakGroups,
    CreateIndex
)
from gscore.models.scorer import (
    TargetScoringModel,
    ADAM_OPTIMIZER,
    EARLY_STOPPING_CB
)
from gscore.osw.connection import (
    OSWConnection
)
from gscore.models.preprocess import (
    preprocess_data,
    STANDARD_SCALAR_PIPELINE
)


def combine_peak_group_data(scored_files, cutoff):


    decoy_free = False

    combined_peak_groups = list()

    for osw_path in scored_files:


        #TODO: Change query names here (fetch decoy vs fetch decoy free)

        peak_groups = fetch_peak_groups(
            host=osw_path,
            query=SelectPeakGroups.FETCH_VOTED_DATA
        )
        
        peak_groups.rerank_groups(
            rerank_keys=['var_xcorr_shape_weighted'], 
            ascending=False
        )

        highest_ranking = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['var_xcorr_shape_weighted'], 
            ascending=False
        )

        if decoy_free:

            low_ranking = list()

            for rank in range(3, 6):

                lower_ranking = peak_groups.select_peak_group(
                    rank=rank,
                    rerank_keys=['var_xcorr_shape_weighted'],
                    ascending=False
                )

                low_ranking.append(lower_ranking)

            lower_ranking = pd.concat(
                low_ranking,
                ignore_index=True
            )

            target_data = denoise_target_labels(
                highest_ranking
            )

            false_target_data = denoise_false_target_labels(
                lower_ranking
            )

        else:
            print("Here")

            target_data = highest_ranking[
                highest_ranking['target'] == 1.0
            ].copy()

            target_data = denoise_target_labels(
                target_data
            )

            decoy_peak_groups = fetch_peak_groups(
                host=osw_path,
                query=SelectPeakGroups.FETCH_DECOY_PEAK_GROUPS
            )

            decoy_peak_groups.rerank_groups(
                rerank_keys=['var_xcorr_shape_weighted'],
                ascending=False
            )

            false_target_data = decoy_peak_groups.select_peak_group(
                rank=1,
                rerank_keys=['var_xcorr_shape_weighted'],
                ascending=False
            )

        false_target_data = resample(
            false_target_data,
            replace=False,
            n_samples=len(target_data),
            random_state=42
        )

        #TODO: Rename these variables and the denoise label function
        denoised_target_labels = pd.concat(
            [
                target_data,
                false_target_data
            ],
            ignore_index=True
        )

        print("Labels: ", denoised_target_labels.target.value_counts())
        
        combined_peak_groups.append(
            denoised_target_labels
        )

    return pd.concat(combined_peak_groups, ignore_index=True)


def denoise_target_labels(peak_groups):

    targets = peak_groups[
        (peak_groups['vote_percentage'] == 1.0) &
        (peak_groups['target'] == 1.0)
    ].copy()

    return targets

def denoise_false_target_labels(peak_groups):

    false_targets = peak_groups[
        #(peak_groups['vote_percentage'] == 0.0) &
        (peak_groups['target'] == 1.0)
    ].copy()

    false_targets['target'] = 0.0

    return false_targets

def main(args, logger):

    scored_files = [input_file.name for input_file in args.input_osw_files]

    logger.info("[INFO] Combining datasets")

    combined_peak_groups = combine_peak_group_data(
        scored_files, 
        args.target_vote_cutoff
    )

    openswath_input = True

    if openswath_input:

        ml_features = [
            col for col in combined_peak_groups.columns
            if col.startswith('var')
        ]

    logger.info("[INFO] Denoising target labels")

    training_data, testing_data = train_test_split(
        combined_peak_groups, 
        test_size=0.1, 
        random_state=42
    )

    print(training_data.target.value_counts())

    scaling_pipeline = STANDARD_SCALAR_PIPELINE

    training_processed, scaling_pipeline = preprocess_data(
        pipeline=scaling_pipeline,
        data=training_data.copy(),
        columns=ml_features,
        train=True,
        return_scaler=True
    )

    save_scaler_path = f"{args.output_directory}/scalers/{args.model_name}.pkl"

    with open(save_scaler_path, 'wb') as pkl:
        pickle.dump(scaling_pipeline, pkl)

    testing_processed = preprocess_data(
        pipeline=scaling_pipeline,
        data=testing_data.copy(),
        columns=ml_features,
    )

    model = TargetScoringModel(
        input_dim=training_processed[ml_features].shape[1:]
    )

    model.compile(loss='binary_crossentropy',
              optimizer=ADAM_OPTIMIZER,
              metrics=['accuracy']
    )

    logger.info("[INFO] Training model")

    history = model.fit(
        training_processed[ml_features], 
        training_processed['target'],
        epochs=50,
        validation_split=0.10,
        callbacks=[EARLY_STOPPING_CB],
        batch_size=32
    )

    logger.info("[INFO] Evaluating model")

    model.evaluate(
        testing_processed[ml_features], 
        testing_processed['target']
    )

    logger.info("[INFO] Saving model")

    model.save(f"{args.output_directory}/{args.model_name}")
