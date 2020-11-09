import pandas as pd

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE,
    CREATE_GHOSTSCORE_TABLE
)
from gscore.models.denoiser import DenoizingClassifier
from gscore.osw.connection import (
    create_table,
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
            {'feature_id': feature_id,
            'vote_percentage': vote}
        )
    
    return record_updates


def add_vote_records(records, osw_path):

    with OSWConnection(osw_path) as conn:

        conn.add_records(
            table_name='ghost_score_table', 
            records=records
        )


def main(args, logger):

    logger.info(f'[INFO] Beginning target label denoising for {args.input_osw_file}')

    logger.debug(f'[DEBUG] Extracting true targets and reranking')

    peak_groups = fetch_peak_groups(
        host=args.input_osw_file,
        query=FETCH_UNSCORED_PEAK_GROUPS_DECOY_FREE
    )

    peak_groups.rerank_groups(
        rerank_keys=['var_xcorr_shape'],
        ascending=False
    )

    logger.debug(f'[DEBUG] Extracting second ranked targets as decoys and reranking')

    second_ranking = peak_groups.select_peak_group(
        rank=2,
        rerank_keys=['var_xcorr_shape'], 
        ascending=False
    )

    second_ranking['target'] = 0.0

    highest_ranking = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['var_xcorr_shape'], 
        ascending=False
    )

    noisey_target_labels = pd.concat(
        [
            highest_ranking,
            second_ranking
        ]
    )

    denoizing_classifier = DenoizingClassifier(
        target_label='target', 
        columns=peak_groups.ml_features
    )

    logger.debug(f'[DEBUG] Fitting denoising classifier {args.num_classifiers} for {args.num_folds}')

    denoizing_classifier.fit(
        data=noisey_target_labels,
        folds=args.num_folds,
        num_classifiers=args.num_classifiers
    )

    logger.debug(f'[DEBUG] Calculating vote percentage')

    noisey_target_labels['vote_percentage'] = denoizing_classifier.vote()

    record_updates = prepare_add_records(noisey_target_labels)

    logger.debug(f'[DEBUG] Adding records to library')

    if args.output_osw_file:

        create_table(
            args.output_osw_file, 
            CREATE_GHOSTSCORE_TABLE
        )

        add_vote_records(
            records=record_updates,
            osw_path=args.output_osw_file
        )
    else:

        create_table(
            args.input_osw_file, 
            CREATE_GHOSTSCORE_TABLE
        )

        add_vote_records(
            records=record_updates,
            osw_path=args.input_osw_file
        )
