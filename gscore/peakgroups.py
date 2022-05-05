from __future__ import annotations

from typing import Dict, Union

import numpy as np
import numpy.typing as npt

from gscore.chromatograms import Chromatogram


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
    chromatograms: Union[Dict[str, Chromatogram], None]
    probability: float
    vote_percentage: float
    true_target_score: float
    true_target_probability: float
    d_score: float
    q_value: float
    scores: Dict[str, float]

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

        self.chromatograms = None

        self.probability = probability
        self.vote_percentage = vote_percentage
        self.true_target_score = true_target_score
        self.true_target_probability = true_target_probability
        self.d_score = d_score
        self.q_value = q_value

        self.scaled_rt_start = 0.0
        self.scaled_rt_apex = 0.0
        self.scaled_rt_end = 0.0

    def __repr__(self) -> str:

        return f"{self.mz=} {self.retention_time=} {self.decoy=} {self.scores=}"

    def get_chromatogram_rt_array(self, interpolated: bool=False, num_rt_steps: int=25) -> npt.NDArray[np.float64]:

        chromatogram = list(self.chromatograms.values())[0]

        if interpolated:

            return chromatogram.interpolated_rt(num_steps=25)

        return chromatogram.rts

    def get_chromatogram_intensity_arrays(self, num_chromatograms: int=6) -> npt.NDArray[np.float64]:

        intensities = np.array([chromatogram.intensities for chromatogram in self.chromatograms.values()])

        if intensities.shape[0] < num_chromatograms:

            difference = num_chromatograms - len(intensities)

            padded_chromatograms = np.zeros((difference, 25), dtype=float)

            intensities = np.concatenate((intensities, padded_chromatograms), axis=0)

        if np.max(intensities) != 0.0:

            intensities = intensities / np.max(intensities)

        return intensities[intensities.mean(axis=1).argsort()]

    def add_score_column(self, key: str, value: float) -> None:

        self.scores[key] = value

    def get_sub_score_column_array(self, include_probability: bool) -> npt.NDArray[np.double]:

        score_values = [score for score in self.scores.values()]

        if include_probability:

            score_values.append(self.probability)

        return np.asarray(score_values, dtype=np.double)
