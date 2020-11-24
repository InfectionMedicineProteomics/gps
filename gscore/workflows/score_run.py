import pickle

import pandas as pd
import numpy as np

from tensorflow import keras

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE,
    CREATE_GHOSTSCORE_TABLE
)
from gscore.models.denoiser import DenoizingClassifier
from gscore.osw.connection import (
    OSWConnection
)

from gscore.workflows.denoise import denoise
from gscore.models.preprocess import preprocess_data


def prepare_add_records(records):

    record_updates = records[
        [
            'feature_id',
            'vote_percentage',
            'd_score',
            'alt_d_score'
        ]
    ]

    feature_ids = list(record_updates['feature_id'])

    votes = list(record_updates['vote_percentage'])

    d_scores = list(record_updates['d_score'])

    alt_d_scores = list(record_updates['alt_d_score'])

    record_updates = list()

    for feature_id, vote, d_score, alt_d_score in zip(feature_ids, votes, d_scores, alt_d_scores):

        record_updates.append(
            {'feature_id': feature_id,
            'vote_percentage': vote,
            'd_score': d_score,
            'alt_d_score': alt_d_score}
        )
    
    return record_updates


def add_vote_records(records, osw_path):

    with OSWConnection(osw_path) as conn:

        conn.add_records(
            table_name='ghost_score_table', 
            records=records
        )


def score(data, columns, logger=None, model_path='', scaler_path=''):

    with open(scaler_path, 'rb') as pkl:
        pipeline = pickle.load(pkl)

    scoring_model = keras.models.load_model(
        model_path
    )

    all_peak_groups = preprocess_data(
        pipeline=pipeline,
        data=data.copy(),
        columns=columns,
    )

    all_peak_groups['d_score'] = scoring_model.predict(
        all_peak_groups[columns]
    ).ravel()

    return all_peak_groups

def main(args, logger):

    logger.info(f'[INFO] Beginning scoring for {args.input_osw_file}')

    logger.debug(f'[DEBUG] Extracting true targets and reranking')

    peak_groups = fetch_peak_groups(
        host=args.input_osw_file,
        query=FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE
    )

    peak_groups.rerank_groups(
        rerank_keys=['var_xcorr_shape_weighted'],
        ascending=False
    )

    logger.debug(f'[DEBUG] Extracting second ranked targets as decoys and reranking')

    low_ranking = list()

    for rank in range(2, 3):

        lower_ranking = peak_groups.select_peak_group(
            rank=rank,
            rerank_keys=['var_xcorr_shape_weighted'], 
            ascending=False
        )

        lower_ranking['target'] = 0.0

        low_ranking.append(lower_ranking)
    
    lower_ranking = pd.concat(
        low_ranking,
        ignore_index=True
    )

    highest_ranking = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['var_xcorr_shape_weighted'], 
        ascending=False
    )

    noisey_target_labels = pd.concat(
        [
            highest_ranking,
            lower_ranking
        ],
        ignore_index=True
    )

    all_peak_groups = denoise(
        training_data=noisey_target_labels,
        peak_groups=peak_groups,
        columns=peak_groups.ml_features,
        logger=logger,
        num_folds=args.num_folds,
        num_classifiers=args.num_classifiers
    )

    all_peak_groups = score(
        data=all_peak_groups,
        columns=peak_groups.ml_features,
        logger=logger,
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )

    all_peak_groups['alt_d_score'] = np.exp(
        all_peak_groups['vote_percentage']
    ) * all_peak_groups['d_score']

    record_updates = prepare_add_records(all_peak_groups)

    logger.debug(f'[DEBUG] Adding records to library')

    if args.output_osw_file:

        pass

    else:

        with OSWConnection(args.input_osw_file) as conn:

            conn.drop_table(
                'ghost_score_table'
            )

            conn.create_table(
                CREATE_GHOSTSCORE_TABLE
            )

            conn.add_records(
                table_name='ghost_score_table', 
                records=record_updates
            )
