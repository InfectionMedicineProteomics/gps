import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

from sklearn.neighbors import KernelDensity


class LabelDistribution:

    def __init__(self, axis_span=()):

        self.model = KernelDensity(
            bandwidth=0.75
        )

        self.x_axis = np.linspace(
            start=axis_span[0],
            stop=axis_span[1],
            num=1000
        )[:, np.newaxis]

    def values(self, data):

        log_density = self.model.score_samples(data)

        return np.exp(log_density)

    def fit(self, data):

        self.model.fit(
            np.array(data).reshape(-1, 1)
        )

    @property
    def max_value(self):
        return self.x_axis[-1]

    @property
    def min_value(self):
        return self.x_axis[0]


class ScoreDistribution:

    def __init__(self, decoy_scores, target_scores, x_axis):

        self.x_axis = x_axis
        self.min_value = x_axis[0]
        self.max_value = x_axis[-1]

        self.null_function = InterpolatedUnivariateSpline(
            x=x_axis,
            y=decoy_scores,
            ext=1
        )

        self.target_function = InterpolatedUnivariateSpline(
            x=x_axis,
            y=target_scores,
            ext=1
        )

    def calc_q_value(self, score):

        if score >= self.max_value:
            return 0.0

        null_area = self.null_function.integral(
            a=score,
            b=self.max_value,
        )

        target_area = self.target_function.integral(
            a=score,
            b=self.max_value
        )

        total_area = null_area + target_area

        try:

            q_value = null_area / total_area

        except ZeroDivisionError as e:
            raise e

        return q_value
