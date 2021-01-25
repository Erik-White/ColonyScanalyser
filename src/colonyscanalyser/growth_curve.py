import warnings
from typing import Any, Union, Optional, Iterable, Dict, List, Tuple
from abc import ABC, abstractmethod
from math import e, exp, log, log10, sqrt
from datetime import timedelta


class GrowthCurve(ABC):
    """
    An abstract class to provide growth curve fitting and parameters
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._growth_curve = None

    @property
    def growth_curve(self) -> "GrowthCurveModel":
        if self._growth_curve is None:
            # Pass the parent instance to allow access to _growth_curve_data
            self._growth_curve = GrowthCurveModel(parent = self)

        return self._growth_curve

    @property
    @abstractmethod
    def _growth_curve_data(self) -> Dict[timedelta, Union[float, List[float]]]:
        """
        A set of growth measurements over time

        Provides data for GrowthCurveModel.fit_curve

        :returns: a dictionary of measurements at time intervals
        """
        raise NotImplementedError("This property must be implemented in a derived class")


class GrowthCurveModel:
    """
    Provides growth curve fitting and parameters for GrowthCurve
    """
    def __init__(self, parent: GrowthCurve):
        if not isinstance(parent, GrowthCurve):
            raise ValueError(f"The enclosing class must be an instance of GrowthCurve, not {type(parent)}")
        self._parent = parent

        self._model = None
        self._growth_rate = None
        self._growth_rate_std = None
        self._carrying_capacity = None
        self._carrying_capacity_std = None
        self._lag_time = None
        self._lag_time_std = None

    @property
    def carrying_capacity(self) -> float:
        """
        The maximal population size, A

        Defined as the asymtote approached by the maximal growth measurement

        :returns: the maximal colony area, in units of log2[area]
        """
        if self._carrying_capacity is None:
            self.fit_curve()

        return self._carrying_capacity

    @property
    def carrying_capacity_std(self) -> float:
        """
        The standard deviation of the maximal population size, A

        :returns: standard deviation of carrying_capacity, in units of log2[area]
        """
        if self._carrying_capacity_std is None:
            self.fit_curve()

        return self._carrying_capacity_std

    @property
    def doubling_time(self) -> timedelta:
        """
        The doubling time at the maximal growth rate

        Defined as ln2 / μmax

        :returns: the minimum time taken for a colony to double in size as a timedelta
        """
        doubling = 0

        if self.growth_rate > 0:
            doubling = log(2) / self.growth_rate

        return timedelta(seconds = doubling)

    @property
    def doubling_time_std(self) -> timedelta:
        """
        The standard deviation of the doubling time at the maximal growth rate

        :returns: standard deviation of doubling_time as a timedelta
        """
        doubling = 0

        if self.growth_rate_std > 0:
            doubling = log(2) / self.growth_rate_std

        return timedelta(seconds = doubling)

    @property
    def data(self) -> Dict[timedelta, Union[float, List[float]]]:
        """
        A set of growth measurements over time

        :returns: a dictionary of measurements at time intervals
        """
        return self._parent._growth_curve_data

    @property
    def growth_rate(self) -> float:
        """
        The maximum specific growth rate, μmax

        Defined as the tangent in the inflection point of the growth curve

        :returns: the maximal growth rate in units of log2[area] / second
        """
        if self._growth_rate is None:
            self.fit_curve()

        return self._growth_rate

    @property
    def growth_rate_std(self) -> float:
        """
        The standard deviation of the maximum specific growth rate, μmax

        :returns: the standard deviation of growth_rate, in units of log2[area] / second
        """
        if self._growth_rate_std is None:
            self.fit_curve()

        return self._growth_rate_std

    @property
    def lag_time(self) -> timedelta:
        """
        The lag time, λ

        Defined as the x-axis intercept of the maximal growth rate (μmax)

        :returns: the lag phase of growth as a timedelta
        """
        if self._lag_time is None:
            self.fit_curve()

        return self._lag_time

    @property
    def lag_time_std(self) -> timedelta:
        """
        The standard deviation of the lag time, λ

        :returns: the lag phase of growth as a timedelta
        """
        if self._lag_time_std is None:
            self.fit_curve()

        return self._lag_time_std

    @property
    def model(self) -> callable:
        """
        A three parameter growth model to be used by fit_curve

        Defaults to the Gompertz function

        :returns: a growth model function
        """
        if self._model is None:
            self._model = self._gompertz

        return self._model

    @model.setter
    def model(self, val: callable):
        self._model = val

    def fit_curve(self, initial_params: List[float] = None):
        """
        Fit a growth model to data

        Growth data must be provided by _growth_curve_data in the implementing class

        :param initial_params: initial estimate of parameters for the growth model
        """
        from statistics import median
        from numpy import errstate, iinfo, intc, isinf, isnan, sqrt, diag, std
        from .utilities import savgol_filter

        timestamps = [timestamp.total_seconds() for timestamp in sorted(self.data.keys())]
        measurements = [val for _, val in sorted(self.data.items())]
        measurements_std = None

        lag_time = 0.0
        lag_time_std = 0.0
        growth_rate = 0.0
        growth_rate_std = 0.0
        carrying_capacity = 0.0
        carrying_capacity_std = 0.0

        # The number of values must be at least the number of parameters
        if len(timestamps) >= 4 and len(measurements) >= 4:
            if all(isinstance(m, Iterable) for m in measurements):
                # Calculate standard deviation
                measurements_std = [std(m, axis = 0) for m in measurements]
                # Use the filtered median
                measurements = [median(val) for val in measurements]
                measurements = savgol_filter(measurements, window = 15, order = 2)

            # Try to estimate initial parameters, if unsuccessful pass None
            # None will result in scipy.optimize.curve_fit using its own default parameters
            if initial_params is None:
                window = 15 if len(timestamps) > 15 else len(timestamps) // 3
                params_estimate = self.estimate_parameters(timestamps, measurements, window = window)
                if params_estimate:
                    initial_params = [min(measurements), *params_estimate]

            # Suppress divide by zero errors caused by zeroes in sigma values
            with errstate(divide = "ignore"):
                results = self._fit_curve(
                    self.model,
                    timestamps,
                    measurements,
                    initial_params = initial_params,
                    sigma = measurements_std,
                    absolute_sigma = True
                )

            if results is not None:
                results, conf = results

                if (not isinf(results).any() and not isnan(results).any()
                        and not (results < 0).any() and not (results >= iinfo(intc).max).any()):
                    _, lag_time, growth_rate, carrying_capacity = results

                    # Calculate standard deviation if results provided
                    if (not isinf(conf).any() and not isnan(conf).any()
                            and not (results < 0).any() and not (conf >= iinfo(intc).max).any()):
                        _, lag_time_std, growth_rate_std, carrying_capacity_std = sqrt(diag(conf.clip(min = 0)))

        self._lag_time = timedelta(seconds = lag_time)
        self._lag_time_std = timedelta(seconds = lag_time_std)
        self._growth_rate = growth_rate
        self._growth_rate_std = growth_rate_std
        self._carrying_capacity = carrying_capacity
        self._carrying_capacity_std = carrying_capacity_std

    @staticmethod
    def estimate_parameters(timestamps: Iterable[float], measurements: Iterable[float], window: int = 10) -> Tuple[float]:
        """
        Estimate the initial parameters for curve fitting

        Lag time:
            Approximates the inflection point in the growth curve as the timestamp where the
            difference in measurements is greater than the mean difference between all measurements,
            plus the standard deviation.

            If the growth rate can be found with linear regression, the intercept of the slope of
            the maximum specific growth rate with the time is taken instead

        Growth rate:
            Approximates the maximum specific growth rate as maximum growth rate measured over a
            sliding window

        Carrying capacity:
            Approximates the asymptote approached by the growth curve as the maximal measurement plus
            the standard deviation of the differences between measurements

        :param timestamps: a collections of time values as floats
        :param measurements: a collection of growth measurements corresponding to timestamps
        :param window: the window size used for finding the maximum growth rate
        :returns: estimation of lag time, growth rate and carrying capacity
        """
        from numpy import diff
        from scipy.stats import linregress

        if not len(timestamps) > 0 or not len(measurements) > 0 or len(timestamps) < window:
            return 0, 0, 0

        if len(timestamps) != len(measurements):
            raise ValueError(
                f"The timestamps ({len(timestamps)} elements) and measurements"
                f" ({len(measurements)} elements) must contain the same number of elements,"
                f" and contain at least as many elements as the window size ({window})"
            )

        diffs = diff(measurements)

        # Carrying capacity
        carrying_capacity = max(measurements) + diffs.std()

        # Lag time and growth rate
        slopes = list()
        for i in range(0, len(timestamps) - window):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # Find the slope at the exponential growth phase over a sliding window
                slope, intercept, *_ = linregress(timestamps[i: i + window], measurements[i: i + window])
                slopes.append((slope, intercept))

        if slopes and max(slopes)[0] > 0:
            growth_rate, intercept = max(slopes)
            lag_time = -intercept / growth_rate
        else:
            # Find the value closest to diffs_std, and then it's index
            diffs_std = diffs.mean() + diffs.std()
            diffs_index = min(diffs, key = lambda x: abs(x - diffs_std))
            inflection = list(diffs).index(diffs_index)
            lag_time = timestamps[inflection]
            growth_rate = max(diffs)

        return lag_time, growth_rate, carrying_capacity

    @staticmethod
    def _gompertz(
        elapsed_time: float,
        initial_size: float,
        lag_time: float,
        growth_rate: float,
        carrying_capacity: float
    ) -> float:
        """
        Parametrized version of the Gompertz function

        From Herricks et al, 2016 doi: 10.1534/g3.116.037044

        :param elapsed_time: time since start
        :param initial_size: initial growth measurement
        :param growth_rate: the maximum specific growth rate, μmax
        :param lag_time: the time at the inflection point in the growth curve
        :param carrying_capacity: the maximal population size, A
        :returns: a value for the colony area at elapsed_time, in units of log2[area]
        """
        from scipy.special import logsumexp

        try:
            return (
                initial_size + carrying_capacity * exp(
                    # scipy.special.logsumexp is used to minimise overflow errors
                    -logsumexp((
                        ((growth_rate * e) / carrying_capacity) * (lag_time - elapsed_time)
                    ) + log10((3 + sqrt(5)) / 2))
                )
            )
        except (OverflowError, ZeroDivisionError):
            return 0

    @staticmethod
    def _fit_curve(
        curve_function: callable,
        timestamps: List[float],
        measurements: List[float],
        initial_params: List[float] = None,
        **kwargs
    ) -> Optional[Tuple[Any]]:
        """
        Uses non-linear least squares to fit a function to data

        timestamps and measurements should be the same length

        :param curve_function: a function to fit to data
        :param timestamps: a list of observation timestamps
        :param measurements: a list of growth observations
        :param initial_params: initial estimate for the parameters of curve_function
        :param kwargs: arguments to pass to scipy.optimize.curve_fit
        :returns: a tuple containing optimal result parameters, or None if no fit could be made
        """
        from scipy.optimize import curve_fit, OptimizeWarning

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", OptimizeWarning)
                return curve_fit(curve_function, timestamps, measurements, p0 = initial_params, **kwargs)
        except RuntimeError:
            return None