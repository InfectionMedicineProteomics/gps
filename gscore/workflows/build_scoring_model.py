#!/usr/bin/env python3
import datetime
import pickle

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

from gscore.workflows.model_single_run import (
    format_model_distribution,
    save_plot
)
from gscore.osw.peakgroups import PeakGroupList

def main(args, logger):

    multi_distributions = list()

    logger.info('starting model building')

    for osw_path in args.input_osw_files:

        osw_path = osw_path.name

        logger.info(f'building model for {osw_path}')

        peak_groups = fetch_peak_groups(
            host=osw_path,
            query=FETCH_SCORED_DATA
        )

        peak_groups.rerank_groups(
            rerank_keys=['alt_d_score'],
            ascending=False
        )

        logger.info('selecting highest ranking peakgroups')

        highest_ranking = peak_groups.select_peak_group(
            rank=1,
            rerank_keys=['alt_d_score'],
            ascending=False
        )

        logger.info('identifying proteotypic peptides')

        proteotypic_peptides = peak_groups.select_proteotypic_peptides(
            rerank_keys=['alt_d_score']
        )

        highest_ranking['peptide_sequence_charge'] = highest_ranking.apply(
            lambda row: '{}_{}'.format(row['peptide_sequence'], row['charge']),
            axis=1
        )

        logger.info('estimating target/false target distributions')

        targets = highest_ranking[
            highest_ranking['vote_percentage'] == 1.0
        ].copy()

        targets = targets.loc[
            targets.peptide_sequence_charge.isin(proteotypic_peptides)
        ].copy()

        targets['target'] = 1.0

        decoys = highest_ranking[
            highest_ranking['vote_percentage'] <= 0.5
        ].copy()

        decoys = decoys.loc[
            decoys.peptide_sequence_charge.isin(proteotypic_peptides)
        ].copy()

        decoys['target'] = 0.0

        model_distribution = pd.concat(
            [
                targets,
                decoys
            ]
        )

        multi_distributions.append(model_distribution)
    
    combined_peak_groups = pd.concat(
        multi_distributions,
        ignore_index=True
    )

    logger.info('building combined distributions')

    filtered_peak_groups = build_false_target_protein_distributions(
        targets=combined_peak_groups[
            combined_peak_groups.target == 1.0
        ].copy(),
        false_targets=combined_peak_groups[
            combined_peak_groups.target == 0.0
        ].copy()
    )

    peak_groups = dict(
        tuple(
            filtered_peak_groups.groupby('protein_accession')
        )
    )

    peak_groups = PeakGroupList(peak_groups)

    highest_scoring_per_protein = peak_groups.select_peak_group(
        rank=1,
        rerank_keys=['alt_d_score'],
        ascending=False
    )

    logger.info('creating scoring distribution')

    score_distribution = ScoreDistribution(
        data=highest_scoring_per_protein
    )

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    save_plot(
        score_distribution=score_distribution,
        plot_name=f'{timestamp}_global_scoring_model'
    )

    logger.info('saving scoring distribution model')

    with open(args.model_output_destination, 'wb') as pkl:
        pickle.dump(score_distribution, pkl)
