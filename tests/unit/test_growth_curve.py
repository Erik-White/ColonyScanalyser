import pytest
from unittest.mock import patch
from datetime import timedelta
from numpy import linspace

from colonyscanalyser.growth_curve import (
    GrowthCurve,
    GrowthCurveModel
)


@pytest.fixture
def timestamps(request):
    yield linspace(0, 3, 50)


@pytest.fixture
def measurements(request, timestamps):
    from numpy import absolute
    from numpy.random import normal

    # Generate a sigmoid curve with a very small amount of noise
    yield sigmoid_model(timestamps) + (0.0025 * absolute(normal(timestamps, scale = 1)))


def sigmoid_model(x, a = 0, b = 0, c = 0, d = 0):
    return 1 - (1 / (1 + x ** 4))


@pytest.fixture
def host(request, timestamps, measurements):
    timestamps = [timedelta(seconds = t) for t in timestamps]

    host = GrowthCurveHost(times = timestamps, signals = measurements)
    host.growth_curve.model = sigmoid_model

    yield host


class GrowthCurveBase(GrowthCurve):
    @property
    def _growth_curve_data(self):
        raise NotImplementedError


class GrowthCurveHost(GrowthCurve):
    def __init__(self, times, signals):
        self.times = times
        self.signals = signals

    @property
    def _growth_curve_data(self):
        return {time: signal for time, signal in zip(self.times, self.signals)}


class TestGrowthCurve:
    class TestInitialize:
        def test_init(self):
            base = GrowthCurveBase()

            assert base.growth_curve._lag_time is None
            assert base.growth_curve._lag_time_std is None
            assert base.growth_curve._growth_rate is None
            assert base.growth_curve._growth_rate_std is None
            assert base.growth_curve._carrying_capacity is None
            assert base.growth_curve._carrying_capacity_std is None
            with pytest.raises(NotImplementedError):
                base._growth_curve_data

        def test_init_host(self):
            host = GrowthCurveHost(times = list(), signals = list())

            assert len(host._growth_curve_data) == 0
            assert host.growth_curve.lag_time.total_seconds() == 0
            assert host.growth_curve.lag_time_std.total_seconds() == 0
            assert host.growth_curve.growth_rate == 0
            assert host.growth_curve.growth_rate_std == 0
            assert host.growth_curve.doubling_time.total_seconds() == 0
            assert host.growth_curve.doubling_time_std.total_seconds() == 0
            assert host.growth_curve.carrying_capacity == 0
            assert host.growth_curve.carrying_capacity_std == 0

        def test_init_model(self):
            with pytest.raises(ValueError):
                GrowthCurveModel(self)

    class TestProperties:
        def test_model(self, host):
            assert host.growth_curve.model == sigmoid_model
            host.growth_curve.model = GrowthCurveModel._gompertz
            assert host.growth_curve.model == GrowthCurveModel._gompertz

        def test_growth_curve_data(self, host, timestamps):
            assert isinstance(host.growth_curve.data, dict)
            assert len(host.growth_curve.data) == len(timestamps)

        def test_lag_time(self, host):
            assert isinstance(host.growth_curve.lag_time, timedelta)
            assert round(host.growth_curve.lag_time.total_seconds(), 1) == 0.5

        def test_lag_time_std(self, host):
            assert isinstance(host.growth_curve.lag_time_std, timedelta)
            assert host.growth_curve.lag_time_std.total_seconds() >= 0

        def test_growth_rate(self, host):
            assert isinstance(host.growth_curve.growth_rate, float)
            assert round(host.growth_curve.growth_rate, 1) == 0.9

        def test_growth_rate_std(self, host):
            assert isinstance(host.growth_curve.growth_rate_std, float)
            assert host.growth_curve.growth_rate_std == 0

        def test_doubling_time(self, host):
            assert isinstance(host.growth_curve.doubling_time, timedelta)
            assert round(host.growth_curve.doubling_time.total_seconds(), 1) == 0.8

        def test_doubling_time_std(self, host):
            host.growth_curve._growth_rate_std = 1

            assert isinstance(host.growth_curve.doubling_time_std, timedelta)
            assert round(host.growth_curve.doubling_time_std.total_seconds(), 1) == 0.7

        def test_carrying_capacity(self, host):
            assert isinstance(host.growth_curve.carrying_capacity, float)
            assert round(host.growth_curve.carrying_capacity, 1) == 1

        def test_carrying_capacity_std(self, host):
            assert isinstance(host.growth_curve.carrying_capacity_std, float)
            assert host.growth_curve.carrying_capacity_std == 0

    class TestMethods():
        def test_fit_growth_curve(self, host):
            host.signals = linspace(1, 5, len(host.signals))
            host.growth_curve.fit_curve(initial_params = [1.2, 0.2, 3.5, max(host.signals)])

            assert host.growth_curve._lag_time
            assert host.growth_curve._lag_time_std.total_seconds() >= 0
            assert host.growth_curve._growth_rate
            assert host.growth_curve._growth_rate_std >= 0
            assert host.growth_curve._carrying_capacity
            assert host.growth_curve._carrying_capacity_std >= 0

        def test_fit_growth_curve_iter(self, host):
            host.signals = [[signal] for signal in host.signals]

            host.growth_curve.fit_curve(initial_params = [0.2, 0.4, 1, 1])

            assert host.growth_curve._lag_time
            assert host.growth_curve._lag_time_std.total_seconds() >= 0
            assert host.growth_curve._growth_rate
            assert host.growth_curve._growth_rate_std >= 0
            assert host.growth_curve._carrying_capacity
            assert host.growth_curve._carrying_capacity_std >= 0

        @patch("colonyscanalyser.growth_curve.GrowthCurveModel._fit_curve")
        def test_fit_growth_curve_std(self, patch, host):
            from numpy import array, float

            # Generate some number for standard deviation
            conf = linspace(1, 16, 16)
            conf = conf.reshape(4, 4)
            host.growth_curve._fit_curve.return_value = (array([1, 1, 1, 1], dtype = float), conf)

            host.growth_curve.fit_curve()

            assert host.growth_curve._lag_time_std.total_seconds() == 2.44949
            assert host.growth_curve._growth_rate_std == 3.3166247903554
            assert host.growth_curve._carrying_capacity_std == 4

        def test_estimate_parameters(self, timestamps, measurements):
            from colonyscanalyser.utilities import round_tuple_floats
            assert GrowthCurveModel.estimate_parameters(timestamps, list()) == (0, 0, 0)
            assert GrowthCurveModel.estimate_parameters(timestamps, measurements, window = len(timestamps) + 1) == (0, 0, 0)
            with pytest.raises(ValueError):
                GrowthCurveModel.estimate_parameters(timestamps, [0])
            assert round_tuple_floats(GrowthCurveModel.estimate_parameters(timestamps, measurements), 1) == (0.5, 1, 1)
            assert GrowthCurveModel.estimate_parameters([0, 0, 0, 0, 0, 0], [0, 1, 3, 6, 10, 15], window = 3) == (
                0, 5, 16.414213562373096
            )

        def test_gompertz(self):
            assert GrowthCurveModel._gompertz(1, 1, 1, 1, 1) == 1.6583785030111096
            assert GrowthCurveModel._gompertz(1, 1, 1, 1, 0) == 0

        def test_curve_fit(self):
            params = list(range(10))

            assert GrowthCurveModel._fit_curve(
                lambda x, y: x * y,
                params,
                list(reversed(params)),
                maxfev = 1
            ) is None