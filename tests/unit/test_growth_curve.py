import pytest
from datetime import timedelta

from colonyscanalyser.growth_curve import (
    GrowthCurve
)


@pytest.fixture(params = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
def timestamps(request):
    yield request.param


@pytest.fixture(params = [[0, 0.2, 0.3, 0.5, 0.8, 1.2, 2.5, 4, 8, 15, 25, 35, 40, 42, 41, 41.5, 41.8, 41.9, 42, 42, 42]])
def measurements(request):
    yield request.param


@pytest.fixture
def growth_curve_child(request, timestamps, measurements):
    timestamps = [timedelta(seconds = t) for t in timestamps]

    yield GrowthCurveChild(times = timestamps, signals = measurements)


class GrowthCurveBase(GrowthCurve):
    @property
    def growth_curve_data(self):
        raise NotImplementedError


class GrowthCurveChild(GrowthCurve):
    def __init__(self, times, signals):
        self.times = times
        self.signals = signals

    @property
    def growth_curve_data(self):
        return {time: signal for time, signal in zip(self.times, self.signals)}


class TestGrowthCurve:
    class TestInitialize:
        def test_init(self):
            growth_curve = GrowthCurveBase()

            assert growth_curve._GrowthCurve__lag_time is None
            assert growth_curve._GrowthCurve__lag_time_std is None
            assert growth_curve._GrowthCurve__growth_rate is None
            assert growth_curve._GrowthCurve__growth_rate_std is None
            assert growth_curve._GrowthCurve__carrying_capacity is None
            assert growth_curve._GrowthCurve__carrying_capacity_std is None
            with pytest.raises(NotImplementedError):
                growth_curve.growth_curve_data()

        def test_init_child(self):
            growth_curve = GrowthCurveChild(times = list(), signals = list())

            assert len(growth_curve.growth_curve_data) == 0
            assert growth_curve.lag_time.total_seconds() == 0
            assert growth_curve.lag_time_std.total_seconds() == 0
            assert growth_curve.growth_rate == 0
            assert growth_curve.growth_rate_std == 0
            assert growth_curve.doubling_time.total_seconds() == 0
            assert growth_curve.doubling_time_std.total_seconds() == 0
            assert growth_curve.carrying_capacity == 0
            assert growth_curve.carrying_capacity_std == 0

    class TestProperties:
        def test_growth_curve_data(self, growth_curve_child, timestamps):
            assert isinstance(growth_curve_child.growth_curve_data, dict)
            assert len(growth_curve_child.growth_curve_data) == len(timestamps)

        def test_lag_time(self, growth_curve_child):
            assert isinstance(growth_curve_child.lag_time, timedelta)
            assert growth_curve_child.lag_time.total_seconds() == 3.318548

        def test_lag_time_std(self, growth_curve_child):
            assert isinstance(growth_curve_child.lag_time_std, timedelta)
            assert growth_curve_child.lag_time_std.total_seconds() == 0

        def test_growth_rate(self, growth_curve_child):
            assert isinstance(growth_curve_child.growth_rate, float)
            assert growth_curve_child.growth_rate == 3.6072727272727265

        def test_growth_rate_std(self, growth_curve_child):
            assert isinstance(growth_curve_child.growth_rate_std, float)
            assert growth_curve_child.growth_rate_std == 0

        def test_doubling_time(self, growth_curve_child):
            assert isinstance(growth_curve_child.doubling_time, timedelta)
            assert growth_curve_child.doubling_time.total_seconds() == 0.192153

        def test_doubling_time_std(self, growth_curve_child):
            growth_curve_child._GrowthCurve__growth_rate_std = 1

            assert isinstance(growth_curve_child.doubling_time_std, timedelta)
            assert growth_curve_child.doubling_time_std.total_seconds() == 0.693147

        def test_carrying_capacity(self, growth_curve_child):
            assert isinstance(growth_curve_child.carrying_capacity, float)
            assert growth_curve_child.carrying_capacity == 45.25146120997929

        def test_carrying_capacity_std(self, growth_curve_child):
            assert isinstance(growth_curve_child.carrying_capacity_std, float)
            assert growth_curve_child.carrying_capacity_std == 0

    class TestMethods():
        def test_fit_growth_curve(self, growth_curve_child):
            from numpy import linspace

            growth_curve_child.signals = linspace(1, 6, len(growth_curve_child.signals))
            growth_curve_child.fit_growth_curve(initial_params = [1, 1, 1, 1])

            assert growth_curve_child._GrowthCurve__lag_time
            assert growth_curve_child._GrowthCurve__lag_time_std.total_seconds() >= 0
            assert growth_curve_child._GrowthCurve__growth_rate
            assert growth_curve_child._GrowthCurve__growth_rate_std >= 0
            assert growth_curve_child._GrowthCurve__carrying_capacity
            assert growth_curve_child._GrowthCurve__carrying_capacity_std >= 0

        def test_estimate_parameters(self, timestamps, measurements):
            assert GrowthCurve.estimate_parameters(timestamps, list()) == (0, 0, 0)
            assert GrowthCurve.estimate_parameters(timestamps, measurements, window = len(timestamps) + 1) == (0, 0, 0)
            with pytest.raises(ValueError):
                GrowthCurve.estimate_parameters(timestamps, [0])
            assert GrowthCurve.estimate_parameters(timestamps, measurements) == (
                3.318548387096774, 3.607272727272727, 45.25146120997929
            )
            assert GrowthCurve.estimate_parameters([0, 0, 0, 0, 0, 0], [0, 1, 3, 6, 10, 15], window = 3) == (
                0, 5, 16.414213562373096
            )

        def test_gompertz(self):
            assert GrowthCurve.gompertz(1, 1, 1, 1, 1) == 1.6583785030111096
            assert GrowthCurve.gompertz(1, 1, 1, 1, 0) == 0

        def test_curve_fit(self):
            params = list(range(10))

            assert GrowthCurve._GrowthCurve__fit_curve(
                lambda x, y: x * y,
                params,
                list(reversed(params)),
                maxfev = 1
            ) is None