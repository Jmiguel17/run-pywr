import numpy as np
import pandas as pd
from pywr.nodes import Storage
from scipy.interpolate import Rbf
from pywr.parameter_property import parameter_property
from pywr.parameters import Parameter, load_parameter, load_parameter_values, IndexParameter


class RectifierParameter(Parameter):

    def __init__(self, model, value, lower_bounds=0.0, upper_bounds=np.inf, **kwargs):
        super(RectifierParameter, self).__init__(model, **kwargs)
        self._value = value
        self.double_size = 1
        self.integer_size = 0
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        if self._value < 0.0:
            self.__values[...] =  0.0
        else:
            self.__values[...] = (self._upper_bounds -  self._lower_bounds) * self._value + self._lower_bounds

    def value(self, ts, scenario_index):
        return self._value

    def set_double_variables(self, values):
        self._value = values[0]

    def get_double_variables(self):
        return np.array([self._value, ], dtype=np.float64)

    def get_double_lower_bounds(self):
        return np.array([-0.75], dtype=np.float64)

    def get_double_upper_bounds(self):
        return np.array([1.0], dtype=np.float64)

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


RectifierParameter.register()


class IndexVariableParameter(IndexParameter):

    def __init__(self, model, value, lower_bounds=0, upper_bounds=1, **kwargs):
        super(IndexVariableParameter, self).__init__(model, **kwargs)
        self._value = round(value)
        self.integer_size = 1
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def calc_values(self, timestep):
        # constant parameter can just set the entire array to one value
        self.__indices[...] = self._value
        self.__values[...] = self._value

    def set_integer_variables(self, values):
        self._value  = values[0]

    def get_integer_variables(self):
        return np.array([self._value, ], dtype=np.int32)

    def get_integer_lower_bounds(self):
        return np.array([self._lower_bounds, ], dtype=np.int32)

    def get_integer_upper_bounds(self):
        return np.array([self._upper_bounds, ], dtype=np.int32)

    def index(self, timestep, scenario_index):
        """Returns the current index"""
        # return index as an integer
        return self._value

    @classmethod
    def load(cls, model, data):
        if "value" in data:
            value = data.pop("value")
        else:
            value = load_parameter_values(model, data)
        parameter = cls(model, value, **data)
        return parameter


IndexVariableParameter.register()


class IndexedArrayParameter(Parameter):
    """Parameter which uses an IndexParameter to index an array of Parameters
    An example use of this parameter is to return a demand saving factor (as
    a float) based on the current demand saving level (calculated by an
    `IndexParameter`).
    Parameters
    ----------
    index_parameter : `IndexParameter`
    params : iterable of `Parameters` or floats
    Notes
    -----
    Float arguments `params` are converted to `ConstantParameter`
    """
    def __init__(self, model, index_parameter, params, **kwargs):
        super().__init__(model, **kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.children.add(index_parameter)

        self.params = []
        for p in params:
            if not isinstance(p, Parameter):
                from pywr.parameters import ConstantParameter
                p = ConstantParameter(model, p)
            self.params.append(p)

        for param in self.params:
            self.children.add(param)
        self.children.add(index_parameter)

    def value(self, timestep, scenario_index):
        """Returns the value of the Parameter at the current index"""
        #index = self.index_parameter.get_index(scenario_index)
        
        index = self.index_parameter.get_integer_variables()[0]
        parameter = self.params[index]
        return parameter.get_value(scenario_index)

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop("index_parameter"))
        try:
            parameters = data.pop("params")
        except KeyError:
            parameters = data.pop("parameters")
        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
        return cls(model, index_parameter, parameters, **data)

        
IndexedArrayParameter.register()


class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, factor=1e6, 
                revenue_per_yield=1, application_efficiency=0.8, conveyance_efficiency=0.7, et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self.area = area
        self.factor = factor
        self.et_factor = et_factor
        self.area_factor = area_factor
        self._et_parameter = None
        self._rainfall_parameter = None
        self.reference_et = reference_et
        self.et_parameter = et_parameter
        self.yield_per_area = yield_per_area
        self._crop_water_factor_parameter = None
        self.revenue_per_yield = revenue_per_yield
        self.rainfall_parameter = rainfall_parameter
        self.conveyance_efficiency = conveyance_efficiency
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    et_parameter = parameter_property("_et_parameter")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
      
        # Calculate crop water requirement

        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (self.area * self.area_factor)

        # Calculate overall efficiency
        efficiency = self.application_efficiency * self.conveyance_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency

        return irrigation_water_requirement/self.factor #To have Mm3/day

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):

        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, **data)


IrrigationWaterRequirementParameter.register()


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pd.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pd.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pd.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pd.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)

    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
        
TransientDecisionParameter.register()