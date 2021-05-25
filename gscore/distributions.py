import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline


class LabelDistribution:

    def __init__(self, data=np.array, axis_span=(), model=None):

        self.x_axis = np.linspace(
            start=axis_span[0],
            stop=axis_span[1],
            num=len(data)
        )

        self.values = model.probability(self.x_axis)

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
