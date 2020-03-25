import pytest
from datetime import timedelta

from colonyscanalyser.growth_curve import (
    GrowthCurve,
    GrowthCurveModel
)


@pytest.fixture(params = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
def timestamps(request):
    yield request.param


@pytest.fixture(params = [[0, 0.2, 0.3, 0.5, 0.8, 1.2, 2.5, 4, 8, 15, 25, 35, 40, 42, 41, 41.5, 41.8, 41.9, 42, 42, 42]])
def measurements(request):
    yield request.param


@pytest.fixture
def host(request, timestamps, measurements):
    timestamps = [timedelta(seconds = t) for t in timestamps]

    yield GrowthCurveHost(times = timestamps, signals = measurements)


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
            def test_model(x):
                return x

            assert host.growth_curve.model == GrowthCurveModel._gompertz
            host.growth_curve.model = test_model
            assert host.growth_curve.model == test_model

        def test_growth_curve_data(self, host, timestamps):
            assert isinstance(host.growth_curve.data, dict)
            assert len(host.growth_curve.data) == len(timestamps)

        def test_lag_time(self, host):
            assert isinstance(host.growth_curve.lag_time, timedelta)
            assert host.growth_curve.lag_time.total_seconds() == 3.318548

        def test_lag_time_std(self, host):
            assert isinstance(host.growth_curve.lag_time_std, timedelta)
            assert host.growth_curve.lag_time_std.total_seconds() == 0

        def test_growth_rate(self, host):
            assert isinstance(host.growth_curve.growth_rate, float)
            assert host.growth_curve.growth_rate == 3.6072727272727265

        def test_growth_rate_std(self, host):
            assert isinstance(host.growth_curve.growth_rate_std, float)
            assert host.growth_curve.growth_rate_std == 0

        def test_doubling_time(self, host):
            assert isinstance(host.growth_curve.doubling_time, timedelta)
            assert host.growth_curve.doubling_time.total_seconds() == 0.192153

        def test_doubling_time_std(self, host):
            host.growth_curve._growth_rate_std = 1

            assert isinstance(host.growth_curve.doubling_time_std, timedelta)
            assert host.growth_curve.doubling_time_std.total_seconds() == 0.693147

        def test_carrying_capacity(self, host):
            assert isinstance(host.growth_curve.carrying_capacity, float)
            assert host.growth_curve.carrying_capacity == 45.25146120997929

        def test_carrying_capacity_std(self, host):
            assert isinstance(host.growth_curve.carrying_capacity_std, float)
            assert host.growth_curve.carrying_capacity_std == 0

    class TestMethods():
        def test_fit_growth_curve(self, host):
            from numpy import linspace

            host.signals = linspace(1, 6, len(host.signals))
            host.growth_curve.fit_curve(initial_params = [1, 1, 1, 1])

            assert host.growth_curve._lag_time
            assert host.growth_curve._lag_time_std.total_seconds() >= 0
            assert host.growth_curve._growth_rate
            assert host.growth_curve._growth_rate_std >= 0
            assert host.growth_curve._carrying_capacity
            assert host.growth_curve._carrying_capacity_std >= 0

        def test_estimate_parameters(self, timestamps, measurements):
            assert GrowthCurveModel.estimate_parameters(timestamps, list()) == (0, 0, 0)
            assert GrowthCurveModel.estimate_parameters(timestamps, measurements, window = len(timestamps) + 1) == (0, 0, 0)
            with pytest.raises(ValueError):
                GrowthCurveModel.estimate_parameters(timestamps, [0])
            assert GrowthCurveModel.estimate_parameters(timestamps, measurements) == (
                3.318548387096774, 3.607272727272727, 45.25146120997929
            )
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