import random
from typing import List, Tuple

import numpy as np


def get_precursor_id_folds(
    precursor_ids: List[str], num_folds: int
) -> List[np.ndarray]:

    random.seed(42)
    random.shuffle(precursor_ids)

    folds = np.array_split(precursor_ids, num_folds)

    return folds


def get_training_data_from_npz(file_path: str) -> Tuple[np.ndarray, np.ndarray]:

    npzfile = np.load(file_path)

    return npzfile["x"], npzfile["y"]


def get_training_data(folds: List[np.ndarray], fold_num: int):

    training_data = list()

    for training_fold_idx, training_ids in enumerate(folds):

        if training_fold_idx != fold_num:

            for training_id in training_ids:

                training_data.append(training_id)

    return training_data


def reformat_distribution_data(peakgroups, score_column):

    scores = list()
    score_labels = list()

    for idx, peakgroup in enumerate(peakgroups):

        scores.append(peakgroup.scores[score_column])

        score_labels.append(peakgroup.target)

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)

    return scores, score_labels


def reformat_data(peakgroups, include_score_columns=False):

    scores = list()
    score_labels = list()
    score_indices = list()

    for idx, peakgroup in enumerate(peakgroups):

        score_array = peakgroup.get_sub_score_column_array(include_score_columns)

        scores.append(score_array)

        score_labels.append([peakgroup.target])

        score_indices.append(peakgroup.idx)

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)
    score_indices = np.array(score_indices, dtype=np.str)

    return scores, score_labels, score_indices

def reformat_chromatogram_data(peakgroups, include_scores: List = ["PROBABILITY"], use_interpolated_chroms=False, use_relative_intensities=False):

    scores = list()
    score_labels = list()
    score_indices = list()
    chromatograms = list()

    skipped_peakgroups = 0

    for idx, peakgroup in enumerate(peakgroups):

        if peakgroup.chromatograms:

            score_array = np.array([peakgroup.scores[score] for score in include_scores])

            scores.append(score_array)

            score_labels.append([peakgroup.target])
            score_indices.append(peakgroup.idx)

            peakgroup_chromatograms = peakgroup.get_chromatogram_intensity_arrays(
                interpolated=use_interpolated_chroms,
                use_relative_intensities=use_relative_intensities
            )

            chromatograms.append(peakgroup_chromatograms.reshape(1, 6, 25))

        else:

            skipped_peakgroups += 1

    if skipped_peakgroups > 0:

        print(f"[WARNING] Skipped {skipped_peakgroups} with no found chromatograms.")

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)
    score_indices = np.array(score_indices, dtype=np.str)
    chromatograms = np.array(chromatograms, dtype=np.float)

    return scores, score_labels, score_indices, chromatograms
