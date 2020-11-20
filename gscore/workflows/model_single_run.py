#!/usr/bin/env python3

import pandas as pd
import numpy as np

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_SCORED_DATA
)
from gscore.models.denoiser import DenoizingClassifier
from gscore.osw.connection import OSWConnection

from gscore.models.preprocess import STANDARD_SCALAR_PIPELINE
from gscore.models.distributions import build_false_target_protein_distributions, ScoreDistribution


def prepare_update_records(records, key_field='ghost_score_id'):

    record_ids = list(records[key_field])

    q_values = list(records['q_value'])

    record_updates = {
        record_id: {
            'm_score': q_value
        }
        for record_id, q_value in zip(record_ids, q_values)
    }

    return record_updates

def update_score_records(records, osw_path):

    with OSWConnection(osw_path) as conn:

        conn.update_records(
            table_name='ghost_score_table',
            key_field='ghost_score_id',
            records=records
        )


def format_model_distribution(data, proteotypic_peptides):

    data['peptide_sequence_charge'] = data.apply(
        lambda row: '{}_{}'.format(row['peptide_sequence'], row['charge']),
        axis=1
    )

    targets = data[
        data['vote_percentage'] == 1.0
    ].copy()

    targets = targets.loc[
        targets.peptide_sequence_charge.isin(proteotypic_peptides)
    ].copy()

    decoys = data[
        data['vote_percentage'] == 0
    ].copy()

    decoys = decoys.loc[
        decoys.peptide_sequence_charge.isin(proteotypic_peptides)
    ].copy()

    model_distribution = build_false_target_protein_distributions(
        targets,
        decoys
    )

    return model_distribution


def save_plot(score_distribution, args):

    import matplotlib.pyplot as plt

    plt.plot(score_distribution.target_kde.x_axis, score_distribution.target_kde.values, lw=2, color='cornflowerblue', linestyle='-')
    plt.plot(score_distribution.null_kde.x_axis, score_distribution.null_kde.values, lw=2, color='red', linestyle='-')
    plt.savefig(f'{args.input_osw_file}.scoring_model.pdf')


def main(args, logger):

    logger.info('[INFO] Starting q_value calculation for single run')
    peak_groups = fetch_peak_groups(
        host=args.input_osw_file, 
        query=FETCH_SCORED_DATA
    )

    peak_groups.rerank_groups(
        rerank_keys=['alt_d_score'], 
        ascending=False
    )

    highest_ranking = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['alt_d_score'], 
        ascending=False
    )

    logger.info('[INFO] Select proteotypic peptides')

    proteotypic_peptides = peak_groups.select_proteotypic_peptides(
        rerank_keys=['alt_d_score']
    )

    logger.info('[INFO] Building score distribution')

    model_distribution = format_model_distribution(
        data=highest_ranking,
        proteotypic_peptides=proteotypic_peptides
    )

    score_distribution = ScoreDistribution(
        data=model_distribution
    )

    save_plot(
        score_distribution=score_distribution, 
        args=args
    )

    all_peak_groups = peak_groups.select_peak_group(
        return_all=True
    )

    logger.info('[INFO] Calculating q-values for all peak-groups')

    all_peak_groups['q_value'] = all_peak_groups['alt_d_score'].apply(
        score_distribution.calc_q_value
    )

    record_updates = prepare_update_records(
        all_peak_groups
    )

    logger.info(f'[INFO] Updaing records in database')

    if args.output_osw_file:
        pass
    
    else:

        update_score_records(
            records=record_updates,
            osw_path=args.input_osw_file
        )
