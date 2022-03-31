from typing import Dict

import numpy as np

from gscore.fdr import ScoreDistribution


class Peptide:

    sequence: str
    modified_sequence: str
    decoy: int
    target: int
    q_value: float
    d_score: float
    probability: float

    def __init__(
        self,
        sequence="",
        modified_sequence: str = "",
        decoy: int = 0,
        q_value: float = 0.0,
        d_score: float = 0.0,
        probability=0.0,
    ):

        self.sequence = sequence
        self.modified_sequence = modified_sequence

        self.decoy = decoy
        self.target = abs(decoy - 1)

        self.q_value = q_value
        self.d_score = d_score
        self.probability = probability

    @property
    def identifier(self):

        return self.modified_sequence


class Peptides:

    peptides: Dict[str, Peptide]

    def __init__(self):

        self.peptides = dict()

    def __contains__(self, item: str):

        return item in self.peptides

    def __setitem__(self, key: str, peptide: Peptide):

        self.peptides[key] = peptide

    def __getitem__(self, key: str) -> Peptide:

        return self.peptides[key]

    def __iter__(self):

        for modified_peptide_sequence, peptide in self.peptides.items():

            yield peptide

    def estimate_pit(self, initial_cutoff: float = 0.01):

        peptides = list(self.peptides.values())

        scores = np.zeros((len(self.peptides.values()),), dtype=np.float64)

        labels = np.zeros((len(self.peptides.values()),), dtype=int)

        for i in range(peptides):

            scores[i] = peptides[i].d_score
            labels[i] = peptides[i].target

        score_distribution = ScoreDistribution()

        score_distribution.fit(
            X=scores,
            y=labels
        )

        q_values = score_distribution.calculate_q_values(scores)

        initial_indices = np.argwhere(q_values >= initial_cutoff)

        passed_labels = labels[initial_indices]

        false_target_counts = passed_labels[passed_labels == 1].shape[0]

        decoy_counts = labels[labels == 0].shape[0]

        self.pit = false_target_counts / decoy_counts

        return self.pit
