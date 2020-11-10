import pickle

import pandas as pd
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_VOTED_DATA
)
from gscore.models.scorer import (
    TargetScoringModel,
    ADAM_OPTIMIZER,
    EARLY_STOPPING_CB
)
from gscore.osw.connection import (
    create_table,
    OSWConnection
)
from gscore.models.preprocess import (
    preprocess_data,
    STANDARD_SCALAR_PIPELINE
)


def combine_peak_group_data(scored_files):

    combined_peak_groups = list()

    for osw_path in scored_files:

        peak_groups = fetch_peak_groups(
            host=osw_path, 
            query=FETCH_VOTED_DATA
        )
        
        peak_groups.rerank_groups(
            rerank_keys=['var_xcorr_shape'], 
            ascending=False
        )

        highest_ranking = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['var_xcorr_shape'], 
            ascending=False
        )
        
        combined_peak_groups.append(
            highest_ranking
        )
    return pd.concat(combined_peak_groups)


def denoise_labels(peak_groups, cutoff):

    targets = peak_groups[
        (peak_groups['vote_percentage'] >= cutoff)
    ].copy()

    decoys = peak_groups[
        (peak_groups['vote_percentage'] < cutoff)
    ].copy()

    decoys['target'] = 0.0

    targets['target'] = 1.0

    target_data = pd.concat([targets, decoys])

    return target_data


def main(args, logger):

    scored_files = [input_file.name for input_file in args.input_osw_files]

    logger.info("[INFO] Combining datasets")

    combined_peak_groups = combine_peak_group_data(scored_files)

    ml_features = [
        col for col in combined_peak_groups.columns
        if col.startswith('var')
    ]

    logger.info("[INFO] Denoising target labels")

    target_data = denoise_labels(
        combined_peak_groups, 
        args.target_vote_cutoff
    )

    training_data, testing_data = train_test_split(
        target_data, 
        test_size=0.1, 
        random_state=42,

    )

    scaling_pipeline = STANDARD_SCALAR_PIPELINE

    training_processed, scaling_pipeline = preprocess_data(
        pipeline=scaling_pipeline,
        data=training_data.copy(),
        columns=ml_features,
        train=True,
        return_scaler=True
    )

    save_scaler_path = f"{args.output_directory}/scalers/standard_scaler.pkl"

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
        epochs=20,
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








    









        
