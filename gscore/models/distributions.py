import numpy as np
import pandas as pd

import matplotlib.pyplot as pyplot

from sklearn.neighbors import KernelDensity
from scipy.interpolate import InterpolatedUnivariateSpline


class LabelDistribution(KernelDensity):

    def __init__(self, kernel='epanechnikov', bandwidth=0.25, data=None):

        #bandwidth = 0.3#len(data) ** (-1. / (1 + 4))

        print(bandwidth)

        super().__init__(kernel=kernel, bandwidth=bandwidth)

        self.fit(data)

        self.x_axis = np.linspace(
            start=0.0,
            stop=data.max(),
            num=len(data)
        )[:, np.newaxis]

        self.probability_densities = self.score_samples(self.x_axis)

        self.values = np.exp(self.probability_densities)

        self.max_value = self.x_axis[-1]


class TempDistribution:

    def __init__(self, data=None, bins=None, distribution_type='alt_d_score'):

        self.data = data
        self.bins = self._set_axis(bins)

        self.max_value = self.bins[-1]

    def _set_axis(self, bins):

        return np.linspace(
            start=0.0,
            stop=np.exp(1.0) * 1.0,
            num=len(bins)
        )[:, np.newaxis].ravel()

    def estimate(self):

        estimates, bins = np.histogram(
            self.data,
            bins=self.bins
        )

        interpolate_values = list()

        for value_idx, value in enumerate(estimates):
            
            if value_idx == 0:

                interpolate_values.append(0.0)
            
            interpolate_values.append(value)

        interpolate_values.append(0.0)

        return interpolate_values


class ScoreDistribution:

    def __init__(self, data, distribution_type='alt_d_score'):


        self.bin_edges = np.histogram_bin_edges(
            data[distribution_type], bins='auto'
        )

        self.combined_axis = self._set_axis()
        
        self.target_kde = TempDistribution(
            data=data[
                data['target'] == 1.0
            ][distribution_type],
            bins=self.bin_edges,
            distribution_type=distribution_type
        )

        self.null_kde = TempDistribution(
            data=data[
                data['target'] == 0.0
            ][distribution_type],
            bins=self.bin_edges,
            distribution_type=distribution_type
        )

        # self.combined_axis = np.array(
        #     data[distribution_type]
        # ).reshape(-1, 1).ravel()

        # self.combined_axis.sort()


        # self.target_kde = LabelDistribution(
        #     data=np.array(
        #         data[
        #             data['target'] == 1.0
        #         ][distribution_type]
        #     ).reshape(-1, 1)
        # )

        # self.null_kde = LabelDistribution(
        #     data=np.array(
        #         data[
        #             data['target'] == 0.0
        #         ][distribution_type]
        #     ).reshape(-1, 1)
        # )

        # self.combined_axis = np.array(
        #     data[distribution_type]
        # ).reshape(-1, 1).ravel()

        

        self.target_values = self.target_kde.estimate()
        self.null_values = self.null_kde.estimate()

        self.target_distribution = InterpolatedUnivariateSpline(
            self.combined_axis,
            self.target_values,
            ext=1
        )

        self.null_distribution = InterpolatedUnivariateSpline(
            self.combined_axis,
            self.null_values,
            ext=1
        )


    def _set_axis(self):

        x_axis = list()

        for edge_num, edge_value in enumerate(self.bin_edges):

            if edge_num == 0:

                x_axis.append(edge_value)

            else:
                interpolated_bin = (edge_value + self.bin_edges[edge_num - 1]) / 2.0

                x_axis.append(interpolated_bin)

        x_axis.append(edge_value)

        return x_axis


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
