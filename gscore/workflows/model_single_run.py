#!/usr/bin/env python3

import pandas as pd
import numpy as np

from gscore.osw.peakgroups import fetch_peak_groups
from gscore.osw.queries import (
    FETCH_SCORED_DATA
)
from gscore.models.denoiser import DenoizingClassifier
from gscore.osw.connection import create_table

from gscore.models.preprocess import STANDARD_SCALAR_PIPELINE
from gscore.models.distributions import build_false_target_protein_distributions, ScoreDistribution



def format_model_distribution(data, proteotypic_peptides):

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

    proteotypic_peptides = peak_groups.select_proteotypic_peptides(
        rerank_keys=['alt_d_score']
    )

    protein_groups = peak_groups.select_protein_groups(
        rerank_keys=['alt_d_score']
    )

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

    all_peak_groups['q_value'] = all_peak_groups['alt_d_score'].apply(
        score_distribution.calc_q_value
    )


