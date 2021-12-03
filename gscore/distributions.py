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

        self.target_data = target_data

        decoy_data = data[
            np.argwhere(labels == 0.0)
        ]

        self.decoy_data = decoy_data

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
            ext=2
        )

        self.decoy_function = InterpolatedUnivariateSpline(
            x=self.x_axis,
            y=self.decoy_scores,
            ext=2
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

            if score > self.decoy_data.max().item():

                decoy_area = 0.0

            else:

                decoy_area = self.decoy_function.integral(
                    a=score,
                    b=self.x_axis[-1].item(),
                )

            if score >= self.target_data.max().item():

                target_area = 1.0

            else:

                target_area = self.target_function.integral(
                    a=score,
                    b=self.x_axis[-1].item()
                )

            target_areas.append(target_area)
            decoy_areas.append(decoy_area)

        target_areas = np.array(target_areas)
        decoy_areas = np.array(decoy_areas)

        total_areas = target_areas + decoy_areas

        q_values = decoy_areas / total_areas

        return q_values
