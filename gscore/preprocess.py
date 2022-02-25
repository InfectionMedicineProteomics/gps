import random
from typing import List, Tuple, Any

import numpy as np
from numpy import ndarray


def get_precursor_id_folds(
    precursor_ids: List[str], num_folds: int
) -> List[np.ndarray]:

    random.seed(42)
    random.shuffle(precursor_ids)

    folds = np.array_split(precursor_ids, num_folds)

    return folds


def get_training_data_from_npz(file_path: str) -> Any:

    npzfile = np.load(file_path)

    return npzfile


def get_training_data(folds: List[np.ndarray], fold_num: int):

    training_data = list()

    for training_fold_idx, training_ids in enumerate(folds):

        if training_fold_idx != fold_num:

            for training_id in training_ids:

                training_data.append(training_id)

    return training_data


def reformat_distribution_data(peakgroups):

    scores = list()
    score_labels = list()

    for idx, peakgroup in enumerate(peakgroups):

        scores.append(peakgroup.d_score)

        score_labels.append(peakgroup.target)

    scores = np.array(scores, dtype=np.float64)
    score_labels = np.array(score_labels, dtype=np.float)

    return scores, score_labels


def reformat_data(peakgroups):

    scores = list()
    score_labels = list()
    score_indices = list()

    for idx, peakgroup in enumerate(peakgroups):

        score_array = peakgroup.get_sub_score_column_array(include_probability=False)

        scores.append(score_array)

        score_labels.append([peakgroup.target])

        score_indices.append(peakgroup.idx)

    scores = np.array(scores, dtype=np.float64)
    scores = scores[:, ~np.all(scores == 0, axis=0)]
    scores = scores[:, ~np.all(np.isnan(scores), axis=0)]

    score_labels = np.array(score_labels, dtype=np.float)
    score_indices = np.array(score_indices, dtype=np.str)

    return scores, score_labels, score_indices


def reformat_chromatogram_data(
    peakgroups, use_relative_intensities=False, training=True
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:

    labels = list()
    indices = list()
    chromatograms = list()
    scores = list()

    skipped_peakgroups = 0

    print("Reformat")

    for idx, peakgroup in enumerate(peakgroups):

        if peakgroup.chromatograms:

            labels.append([peakgroup.target])

            indices.append(peakgroup.idx)

            scores.append(peakgroup.get_sub_score_column_array(include_probability=False))

            peakgroup_chromatograms = peakgroup.get_chromatogram_intensity_arrays(
                use_relative_intensities=use_relative_intensities
            )

            chromatograms.append(peakgroup_chromatograms.reshape(1, 6, 25))

        else:

            if not training:

                labels.append([peakgroup.target])

                indices.append(peakgroup.idx)

                scores.append(peakgroup.get_sub_score_column_array(include_probability=False))

                peakgroup_chromatograms = np.zeros((1, 6, 25), dtype=float)

                chromatograms.append(peakgroup_chromatograms)

            skipped_peakgroups += 1

    if skipped_peakgroups > 0:

        if training:

            print(
                f"[WARNING] {skipped_peakgroups} peakgroups with no found chromatograms found."
            )

        else:

            print(
                f"[WARNING] {skipped_peakgroups} peakgroups with no found chromatograms found. Chromatograms set to 0 for scoring"
            )

    scores = np.array(scores, dtype=np.float64)
    scores = scores[:, ~np.all(scores == 0, axis=0)]
    scores = scores[:, ~np.all(np.isnan(scores), axis=0)]
    scores_array = np.array(scores, dtype=np.float64)

    label_array = np.array(labels, dtype=np.float64)
    indice_array = np.array(indices, dtype=str)
    chromatogram_array = np.array(chromatograms, dtype=np.float64)

    return label_array, indice_array, chromatogram_array, scores_array
