import numpy as np
import pandas as pd

import matplotlib.pyplot as pyplot

from sklearn.neighbors import KernelDensity
from scipy.interpolate import InterpolatedUnivariateSpline

from gscore.osw.peakgroups import PeakGroupList


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

    @property
    def max_value(self):
        return self.x_axis[-1]

    @property
    def min_value(self):
        return self.x_axis[0]


class TempDistribution:

    def __init__(self, data=None, bins=None):

        self.data = data
        self.bins = self._set_axis(
            np.histogram_bin_edges(
                data, bins='auto'
            )
        )

        self.max_value = data.max()

        self.min_value = data.min()

    def _set_axis(self, bins):

        return np.linspace(
            start=0.0,
            stop=np.exp(1.0) * 1.0,
            num=len(bins)
        )[:, np.newaxis].ravel()

    def _set_bin_means(self, bin_edges):

        x_axis = list()

        for edge_num, edge_value in enumerate(bin_edges):

            if edge_num == 0:

                x_axis.append(edge_value)

            else:
                interpolated_bin = (edge_value + bin_edges[edge_num - 1]) / 2.0

                x_axis.append(interpolated_bin)

        x_axis.append(edge_value)

        return x_axis


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

        bins = self._set_bin_means(bins)

        return interpolate_values, bins


class ScoreDistribution:

    def __init__(self, data, distribution_type='alt_d_score'):


        self.bin_edges = np.histogram_bin_edges(
            data[distribution_type], bins='auto'
        )
        
        self.target_kde = LabelDistribution(
            data=np.asarray(
                data[
                    data['target'] == 1.0
                ][distribution_type]
            ).reshape(-1, 1),
        )

        self.null_kde = LabelDistribution(
            data=np.asarray(
                data[
                    data['target'] == 0.0
                ][distribution_type]
            ).reshape(-1, 1),
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

        

        self.target_values, self.target_axis = self.target_kde.values, self.target_kde.x_axis
        self.null_values, self.null_axis = self.null_kde.values, self.null_kde.x_axis

        self.target_distribution = InterpolatedUnivariateSpline(
            self.target_axis,
            self.target_values,
            ext=1
        )

        self.null_distribution = InterpolatedUnivariateSpline(
            self.null_axis,
            self.null_values,
            ext=1
        )

    def calc_q_value(self, prob_score):

        if prob_score <= self.target_kde.min_value:

            return 1.0

        elif prob_score >= self.null_kde.max_value:

            return 0.0

        elif prob_score >= self.target_kde.max_value:

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

            try:

                q_value = null_area / total_area

            except ZeroDivisionError as e:
                print(q_value)
                raise e

        return q_value
    

def build_false_target_protein_distributions(
    model_data=None,
    protein_column='protein_accession'
):
    
    peak_groups = dict(
        tuple(
            model_data.groupby(protein_column)
        )
    )

    peak_groups = PeakGroupList(peak_groups)

    false_target_protein_groups = list()

    target_protein_groups = list()

    for peak_group in peak_groups.peak_groups:

        peak_group = peak_group.peak_group

        false_peak_groups = peak_group[
            peak_group['vote_percentage'] <= 0.5
        ].copy()

        true_peak_groups = peak_group[
            peak_group['vote_percentage'] > 0.5
        ].copy()

        if len(true_peak_groups) > len(false_peak_groups):

            target_protein_groups.append(peak_group)

        elif len(true_peak_groups) < len(false_peak_groups):
            
            false_target_protein_groups.append(peak_group)

    targets = pd.concat(
        target_protein_groups,
        ignore_index=True
    )

    false_targets = pd.concat(
        false_target_protein_groups,
        ignore_index=True
    )

    false_targets['target'] = 0.0

    combined = pd.concat(
        [
            targets,
            false_targets
        ]
    )

    return combined
