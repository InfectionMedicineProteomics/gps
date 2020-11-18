import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from scipy.interpolate import InterpolatedUnivariateSpline


class LabelDistribution(KernelDensity):

    def __init__(self, kernel='epanechnikov', bandwidth=0.1, data=None):

        super().__init__(kernel=kernel, bandwidth=bandwidth)

        self.fit(data)

        self.x_axis = np.linspace(
            start=data.min(),
            stop=data.max(),
            num=len(data)
        )[:, np.newaxis]

        self.probability_densities = self.score_samples(self.x_axis)

        self.values = np.exp(self.probability_densities)

        self.max_value = self.x_axis[-1]


class ScoreDistribution:

    def __init__(self, data, distribution_type='alt_d_score'):

        self.target_kde = LabelDistribution(
            data=np.array(
                data[
                    data['target'] == 1.0
                ][distribution_type]
            ).reshape(-1, 1)
        )

        self.null_kde = LabelDistribution(
            data=np.array(
                data[
                    data['target'] == 0.0
                ][distribution_type]
            ).reshape(-1, 1)
        )

        self.combined_axis = np.array(
            data[distribution_type]
        ).reshape(-1, 1).ravel()

        self.combined_axis.sort()

        self.target_distribution = InterpolatedUnivariateSpline(
            self.target_kde.x_axis,
            self.target_kde.values,
            ext=1
        )

        self.null_distribution = InterpolatedUnivariateSpline(
            self.null_kde.x_axis,
            self.null_kde.values,
            ext=1
        )

    def calc_q_value(self, prob_score):

        if prob_score <= self.combined_axis[0]:

            return 1.0

        elif prob_score >= self.combined_axis[-1]:

            return 0.0

        else:

            null_area = self.null_distribution.integral(
                a=prob_score,
                b=self.null_kde.max_value,
            )

            target_area = self.target_distribution.integral(
                a=prob_score,
                b=self.target_kde.max_value
            )

            total_area = null_area + target_area

            return null_area / total_area
    

def build_false_target_protein_distributions(
    targets, 
    false_targets, 
    protein_column='protein_accession'
):
    
    target_proteins = set(targets[protein_column])

    false_target_proteins = set(false_targets[protein_column])

    protein_overlap = list(
        target_proteins.intersection(
            false_target_proteins
        )
    )

    small_decoys = false_targets.loc[
        ~false_targets[protein_column].isin(protein_overlap)
    ].copy()

    small_decoys['target'] = 0.0

    small_targets = targets.loc[
        ~targets[protein_column].isin(protein_overlap)
    ].copy()

    small_targets['target'] = 1.0

    combined = pd.concat(
        [
            small_targets,
            small_decoys
        ]
    )

    return combined
