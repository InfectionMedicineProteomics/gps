import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

from sklearn.neighbors import KernelDensity

from gscore.utils import ml

# class LabelDistribution:
#
#     def __init__(self, axis_span=()):
#
#         self.model = KernelDensity(
#             bandwidth=0.75
#         )
#
#         self.x_axis = np.linspace(
#             start=axis_span[0],
#             stop=axis_span[1],
#             num=1000
#         )[:, np.newaxis]
#
#     def values(self, data):
#
#         log_density = self.model.score_samples(data)
#
#         return np.exp(log_density)
#
#     def fit(self, data):
#
#         self.model.fit(
#             np.array(data).reshape(-1, 1)
#         )
#
#     @property
#     def max_value(self):
#         return self.x_axis[-1]
#
#     @property
#     def min_value(self):
#         return self.x_axis[0]


class ScoreDistribution:

    x_axis: np.ndarray
    target_model: KernelDensity
    decoy_model: KernelDensity
    target_function: InterpolatedUnivariateSpline
    decoy_function: InterpolatedUnivariateSpline

    def __init__(self):

        self.target_model = KernelDensity(bandwidth=0.5, kernel="epanechnikov")
        self.decoy_model = KernelDensity(bandwidth=0.5, kernel="epanechnikov")

    def fit(self, data: np.ndarray, labels: np.ndarray):

        self.x_axis = np.linspace(
            start=data.min(),
            stop=data.max(),
            num=1000
        )[:, np.newaxis]

        target_data = data[
            np.argwhere(labels == 1.0)
        ]

        decoy_data = data[
            np.argwhere(labels == 0.0)
        ]

        self.target_model.fit(
            target_data
        )

        self.decoy_model.fit(
            decoy_data
        )

        self.target_scores = self.score(model='target')
        self.decoy_scores = self.score(model='decoy')

        self.target_function = InterpolatedUnivariateSpline(
            x=self.x_axis,
            y=self.target_scores,
            ext=0
        )

        self.decoy_function = InterpolatedUnivariateSpline(
            x=self.x_axis,
            y=self.decoy_scores,
            ext=0
        )


    def score(self, model: str):

        if model == "target":

            log_density = self.target_model.score_samples(self.x_axis)

        else:

            log_density = self.decoy_model.score_samples(self.x_axis)

        return np.exp(log_density)


    def calculate_q_vales(self, scores: np.ndarray) -> np.ndarray:

        target_areas = []
        decoy_areas = []

        for score in scores:

            decoy_area = self.decoy_function.integral(
                a=score,
                b=self.x_axis[-1],
            )

            target_area = self.target_function.integral(
                a=score,
                b=self.x_axis[-1]
            )

            target_areas.append(target_area)
            decoy_areas.append(decoy_area)

        target_areas = np.array(target_areas)
        decoy_areas = np.array(decoy_areas)

        total_areas = target_areas + decoy_areas

        q_values = decoy_areas / total_areas

        return q_values


def calculate_q_values(precursors, sort_key: str, use_decoys: bool = True):

    target_peakgroups = precursors.get_target_peakgroups_by_rank(
        rank=1,
        score_key=sort_key,
        reverse=True
    )

    if use_decoys:

        decoy_peakgroups = precursors.get_decoy_peakgroups(
            sort_key=sort_key
        )

    else:

        decoy_peakgroups = precursors.get_target_peakgroups_by_rank(
            rank=2,
            score_key=sort_key,
            reverse=True
        )

    modelling_peakgroups = target_peakgroups + decoy_peakgroups

    scores, labels = ml.reformat_distribution_data(
        modelling_peakgroups,
        score_column=sort_key
    )

    score_distribution = ScoreDistribution()

    score_distribution.fit(
        scores,
        labels
    )

    q_values = score_distribution.calculate_q_vales(scores)

    for idx, peakgroup in enumerate(modelling_peakgroups):

        peakgroup.scores['q_value'] = q_values[idx]

    return precursors




