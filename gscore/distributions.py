import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.interpolate import InterpolatedUnivariateSpline


class LabelDistribution(KernelDensity):

    def __init__(self, kernel='gaussian', data=None, axis_span=()):

        super().__init__(kernel=kernel)

        self.fit(data)

        self.x_axis = np.linspace(
            start=data.min() - 10,
            stop=data.max() + 10,
            num=len(data)
        )

        self.probability_densities = self.score_samples(
            self.x_axis.reshape(-1, 1)
        )

        self.values = np.exp(self.probability_densities)

    @property
    def max_value(self):
        return self.x_axis[-1]

    @property
    def min_value(self):
        return self.x_axis[0]


class ScoreDistribution:

    def __init__(self, null_distribution=None, target_distribution=None):

        self.null_distribution = null_distribution
        self.target_distribution = target_distribution

        self.null_function = InterpolatedUnivariateSpline(
            x=null_distribution.x_axis,
            y=null_distribution.values,
            ext=1
        )

        self.target_function = InterpolatedUnivariateSpline(
            x=target_distribution.x_axis,
            y=target_distribution.values,
            ext=1
        )

    def calc_q_value(self, score):

        if score >= self.target_distribution.max_value:
            return 0.0

        null_area = self.null_function.integral(
            a=score,
            b=self.null_distribution.max_value,
        )

        target_area = self.target_function.integral(
            a=score,
            b=self.target_distribution.max_value
        )

        total_area = null_area + target_area

        try:

            q_value = null_area / total_area

        except ZeroDivisionError as e:
            raise e

        return q_value
