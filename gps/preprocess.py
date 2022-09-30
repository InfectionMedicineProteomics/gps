import random
from typing import List, Tuple, Any

import numpy as np

from gps.peakgroups import PeakGroup


def get_precursor_id_folds(
    precursor_ids: List[str], num_folds: int
) -> List[np.ndarray]:

    random.seed(42)
    random.shuffle(precursor_ids)

    folds: List[np.ndarray] = np.array_split(precursor_ids, num_folds)

    return folds


def get_training_data_from_npz(file_path: str) -> Any:

    npzfile = np.load(file_path)

    return npzfile


def get_training_data(folds: List[np.ndarray], fold_num: int) -> List[str]:

    training_data = list()

    for training_fold_idx, training_ids in enumerate(folds):

        if training_fold_idx != fold_num:

            for training_id in training_ids:

                training_data.append(training_id)

    return training_data


def reformat_distribution_data(
    peakgroups: List[PeakGroup],
) -> Tuple[np.ndarray, np.ndarray]:

    scores = list()
    score_labels = list()

    for idx, peakgroup in enumerate(peakgroups):

        scores.append(peakgroup.d_score)

        score_labels.append(peakgroup.target)

    return np.array(scores, dtype=np.float64), np.array(score_labels, dtype=int)


def reformat_true_target_scores(
    peakgroups: List[PeakGroup],
) -> Tuple[np.ndarray, np.ndarray]:

    scores = list()
    score_labels = list()

    for idx, peakgroup in enumerate(peakgroups):

        scores.append(peakgroup.true_target_score)

        score_labels.append(peakgroup.target)

    return np.array(scores, dtype=np.float64), np.array(score_labels, dtype=float)


def reformat_data(
    peakgroups: List[PeakGroup],
    use_only_spectra_scores: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    scores = list()
    score_labels = list()
    score_indices = list()

    for idx, peakgroup in enumerate(peakgroups):

        score_array = peakgroup.get_sub_score_column_array(
            include_probability=False, use_only_spectra_scores=use_only_spectra_scores
        )

        scores.append(score_array)

        score_labels.append([peakgroup.target])

        score_indices.append(peakgroup.idx)

    array_scores = np.array(scores, dtype=np.float64)
    array_scores = array_scores[:, ~np.all(array_scores == 0, axis=0)]
    array_scores = array_scores[:, ~np.all(np.isnan(array_scores), axis=0)]

    return (
        array_scores,
        np.array(score_labels, dtype=float),
        np.array(score_indices, dtype=str),
    )


def get_probability_vector(peakgroups: List[PeakGroup]) -> np.ndarray:

    target_probabilities = []

    for idx, peakgroup in enumerate(peakgroups):

        target_probabilities.append(peakgroup.true_target_probability)

    return np.array(target_probabilities, dtype=np.float64)


def reformat_chromatogram_data(
    peakgroups: List[PeakGroup], training: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

            scores.append(
                peakgroup.get_sub_score_column_array(include_probability=False)
            )

            peakgroup_chromatograms = peakgroup.get_chromatogram_intensity_arrays()

            chromatograms.append(peakgroup_chromatograms.reshape(1, 6, 25))

        else:

            if not training:

                labels.append([peakgroup.target])

                indices.append(peakgroup.idx)

                scores.append(
                    peakgroup.get_sub_score_column_array(include_probability=False)
                )

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

    scores_array = np.array(scores, dtype=np.float64)
    scores_array = scores_array[:, ~np.all(scores_array == 0, axis=0)]
    scores_array = scores_array[:, ~np.all(np.isnan(scores_array), axis=0)]

    label_array = np.array(labels, dtype=np.float64)
    indice_array = np.array(indices, dtype=str)
    chromatogram_array = np.array(chromatograms, dtype=np.float64)

    return label_array, indice_array, chromatogram_array, scores_array
