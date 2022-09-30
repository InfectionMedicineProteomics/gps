from __future__ import annotations

from typing import Dict, Union, List

import numpy as np


from gps.chromatograms import Chromatogram


class PeakGroup:

    ghost_score_id: str
    idx: str
    mz: float
    retention_time: float
    intensity: float
    decoy: int
    target: int
    delta_rt: float
    start_rt: float
    end_rt: float
    chromatograms: Dict[str, Chromatogram]
    probability: float
    vote_percentage: float
    true_target_score: float
    true_target_probability: float
    d_score: float
    q_value: float
    scores: Dict[str, float]
    chromatogram_prediction: float
    chromatogram_score: float
    peakgroup_prediction: float
    peakgroup_score: float
    top_scoring: Union[bool, None]

    def __init__(
        self,
        ghost_score_id: str = "",
        idx: str = "",
        mz: float = 0.0,
        rt: float = 0.0,
        intensity: float = 0.0,
        decoy: int = 0,
        delta_rt: float = 0.0,
        start_rt: float = 0.0,
        end_rt: float = 0.0,
        probability: float = 0.0,
        vote_percentage: float = 0.0,
        d_score: float = 0.0,
        q_value: float = 0.0,
        true_target_score: float = 0.0,
        true_target_probability: float = 0.0,
        scores: Union[Dict[str, float], None] = None,
        chromatogram_prediction: float = 0,
        chromatogram_score: float = 0.0,
        peakgroup_prediction: float = 0,
        peakgroup_score: float = 0.0,
    ):

        if scores is None:
            self.scores = dict()

        else:

            self.scores = scores

        self.ghost_score_id = ghost_score_id
        self.idx = idx

        self.retention_time = rt
        self.intensity = intensity

        self.mz = mz
        self.intensity = intensity

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.delta_rt = delta_rt
        self.start_rt = start_rt
        self.end_rt = end_rt

        self.chromatograms = dict()

        self.probability = probability
        self.vote_percentage = vote_percentage
        self.true_target_score = true_target_score
        self.true_target_probability = true_target_probability
        self.d_score = d_score
        self.q_value = q_value

        self.scaled_rt_start = 0.0
        self.scaled_rt_apex = 0.0
        self.scaled_rt_end = 0.0

        self.chromatogram_prediction = chromatogram_prediction
        self.chromatogram_score = chromatogram_score
        self.peakgroup_prediction = peakgroup_prediction
        self.peakgroup_score = peakgroup_score

        self.top_scoring = None

    def __repr__(self) -> str:

        return f"{self.mz=} {self.retention_time=} {self.decoy=} {self.scores=}"

    def get_chromatogram_rt_array(
        self, interpolated: bool = False, num_rt_steps: int = 25
    ) -> np.ndarray:

        chromatogram = list(self.chromatograms.values())[0]

        if interpolated:

            return chromatogram.interpolated_rt(num_steps=25)

        return chromatogram.rts

    def get_chromatogram_intensity_arrays(
        self, num_chromatograms: int = 6
    ) -> np.ndarray:

        intensities = np.array(
            [chromatogram.intensities for chromatogram in self.chromatograms.values()]
        )

        if intensities.shape[0] < num_chromatograms:

            difference = num_chromatograms - len(intensities)

            padded_chromatograms = np.zeros((difference, 25), dtype=float)

            intensities = np.concatenate((intensities, padded_chromatograms), axis=0)

        if np.max(intensities) != 0.0:

            intensities = intensities / np.max(intensities)

        return intensities[intensities.mean(axis=1).argsort()]

    def add_score_column(self, key: str, value: float) -> None:

        self.scores[key] = value

    def get_score_columns(self, flagged_columns: List[str]) -> Dict[str, float]:

        scores = {
            score_name: score_value
            for score_name, score_value in self.scores.items()
            if score_name not in flagged_columns
        }

        return scores

    def get_sub_score_column_array(
        self, include_probability: bool, use_only_spectra_scores: bool = False
    ) -> np.ndarray:

        if use_only_spectra_scores:

            columns = {
                "var_library_dotprod",
                "var_library_sangle",
                "var_library_manhattan",
                "var_library_rootmeansquare",
                "var_library_rmsd",
                "var_yseries_score",
                "var_bseries_score",
                "var_massdev_score_weighted",
                "var_isotope_overlap_score",
                "var_library_corr",
                "var_isotope_correlation_score",
                "var_massdev_score",
            }

            score_values = [
                score
                for column, score in self.scores.items()
                if column.lower() in columns
            ]

        else:

            score_values = [score for score in self.scores.values()]

        if include_probability:

            score_values.append(self.probability)

        return np.asarray(score_values, dtype=np.double)

    def get_sub_score_column_names(self) -> List[str]:

        return [col_name for col_name in self.scores.keys()]
