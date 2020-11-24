import pandas as pd

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE,
    CREATE_GHOSTSCORE_TABLE
)
from gscore.models.denoiser import DenoizingClassifier
from gscore.osw.connection import (
    OSWConnection
)


def prepare_add_records(records):

    record_updates = records[
        [
            'feature_id',
            'vote_percentage'
        ]
    ]

    feature_ids = list(record_updates['feature_id'])

    votes = list(record_updates['vote_percentage'])

    record_updates = list()

    for feature_id, vote in zip(feature_ids, votes):

        record_updates.append(
            {
                'feature_id': feature_id,
                'vote_percentage': vote
            }
        )
    
    return record_updates


def denoise(training_data, peak_groups, columns, logger=None, num_folds=10, num_classifiers=500):

    denoizing_classifier = DenoizingClassifier(
        target_label='target', 
        columns=columns
    )

    if logger:
        logger.debug(f'[DEBUG] Fitting denoising classifier {num_classifiers} for {num_folds}')

    denoizing_classifier.fit(
        data=training_data,
        folds=num_folds,
        num_classifiers=num_classifiers
    )

    if logger:
        logger.debug(f'[DEBUG] Calculating vote percentage')

    all_peak_groups = peak_groups.select_peak_group(
        return_all=True
    )

    all_peak_groups['vote_percentage'] = denoizing_classifier.vote(
        data=all_peak_groups,
        match_index='transition_group_id'
    )

    return all_peak_groups


def main(args, logger):

    logger.info(f'[INFO] Beginning target label denoising for {args.input_osw_file}')

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

    second_ranking = peak_groups.select_peak_group(
        rank=2,
        rerank_keys=['var_xcorr_shape_weighted'], 
        ascending=False
    )

    second_ranking['target'] = 0.0

    highest_ranking = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['var_xcorr_shape_weighted'], 
        ascending=False
    )

    noisey_target_labels = pd.concat(
        [
            highest_ranking,
            second_ranking
        ]
    )

    all_peak_groups = denoise(
        training_data=noisey_target_labels,
        peak_groups=peak_groups,
        columns=peak_groups.ml_features,
        logger=logger,
        num_folds=args.num_folds,
        num_classifiers=args.num_classifiers
    )

    record_updates = prepare_add_records(all_peak_groups)

    logger.debug(f'[DEBUG] Adding records to library')

    if args.output_osw_file:

        pass

    else:

        with OSWConnection(args.input_osw_file) as conn:

            conn.create_table(
                CREATE_GHOSTSCORE_TABLE
            )

            conn.add_records(
                table_name='ghost_score_table', 
                records=record_updates
            )
