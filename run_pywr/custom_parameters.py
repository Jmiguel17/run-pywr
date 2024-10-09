import numpy as np
import pandas as pd

from pywr.recorders import *
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


class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, conveyance_efficiency, application_efficiency, 
                 factor=1e6, revenue_per_yield=1,  et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self._area = None
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
        self._conveyance_efficiency = None
        self.conveyance_efficiency = conveyance_efficiency
        self._application_efficiency = None
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    et_parameter = parameter_property("_et_parameter")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")
    conveyance_efficiency = parameter_property("_conveyance_efficiency")
    application_efficiency = parameter_property("_application_efficiency")
    area = parameter_property("_area")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
        conv_efficiency = self.conveyance_efficiency.get_value(scenario_index)
        app_efficiency = self.application_efficiency.get_value(scenario_index)
        area_ = self.area.get_value(scenario_index)
      
        # Calculate crop water requirement
        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (area_ * self.area_factor)

        # Calculate overall efficiency
        efficiency = app_efficiency * conv_efficiency

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

        attribute_list = ["conveyance_efficiency", "application_efficiency", "area", "reference_et", "yield_per_area"]
        attributes = {}

        for attribute in attribute_list:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    attributes[attribute] = data.pop(attribute)
                else:
                    attributes[attribute] = load_parameter(model, data.pop(attribute))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, attributes["area"], attributes["reference_et"], 
                   attributes["yield_per_area"], attributes["conveyance_efficiency"], attributes["application_efficiency"], **data)


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


class RollingNodeFlowRecorder(NodeRecorder):
    """ Records the mean flow of a node for the previous N timesteps (a window).

    This recorder is different to `RollingMeanFlowNodeRecorder` because for the timesteps lower that the window
    we save a value passed in the recorder definition

    Parameters
    ----------

    mode : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    window : int
        The number of timesteps to calculate the flow rolling mean
    hist_rolling : int
        The value to be saved when the timestep is lower that the window
    name : str (optional)
        The name of the recorder

    """

    def __init__(self, model, node, window=None, days=None, hist_rolling=None, name=None, **kwargs):
        super(RollingNodeFlowRecorder, self).__init__(model, node, name=name, **kwargs)
        # self.model = model

        if not window and not days:
            raise ValueError("Either `window` or `days` must be specified.")
        if window:
            self.window = int(window)
        else:
            self.window = 0
        if days:
            self.days = int(days)
        else:
            self.days = 0

        self._data = None
        self.position = 0

        if not hist_rolling:
            raise ValueError("An `hist_rolling` must be specified.")
        else:
            self.hist_rolling = hist_rolling

    def setup(self):
        super(RollingNodeFlowRecorder, self).setup()
        self._data = np.empty([len(self.model.timestepper), len(self.model.scenarios.combinations)])

        if self.days > 0:
            try:
                self.window = self.days // self.model.timestepper.delta
            except TypeError:
                raise TypeError('A rolling window defined as a number of days is only valid with daily time-steps.')
        if self.window == 0:
            raise ValueError("window property of MeanFlowRecorder is less than 1.")

        self._memory = np.zeros([len(self.model.scenarios.combinations), self.window])

    def reset(self):
        super(RollingNodeFlowRecorder, self).reset()
        self.position = 0
        self._memory[:, :] = 0
        self._data[:, :] = 0.0

    def after(self):

        # Save today's flow
        for i in range(0, self._memory.shape[0]):
            self._memory[i, self.position] = self.node.flow[i]

        # Calculate the mean flow
        timestep = self.model.timestepper.current
        if timestep.index < self.window:
            n = timestep.index + 1
        else:
            n = self.window

        # Save the mean flow
        if timestep.index < self.window:
            self._data[int(timestep.index), :] = self.hist_rolling
        else:
            mean_flow = np.mean(self._memory[:, 0:n], axis=1)
            self._data[int(timestep.index), :] = mean_flow

        # Prepare for the next timestep
        self.position += 1
        if self.position >= self.window:
            self.position = 0

    @property
    def data(self):
        return np.array(self._data, dtype=np.float64)

    def to_dataframe(self):
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(data=self.data, index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        name = data.get("name")
        # node = model.nodes[data["node"]] # for the new version of pywr
        node = model._get_node_from_ref(model, data.pop("node"))

        if "hist_rolling" in data:
            hist_rolling = data["hist_rolling"]
        else:
            hist_rolling = None

        if "window" in data:
            window = int(data["window"])
        else:
            window = None

        if "days" in data:
            days = int(data["days"])
        else:
            days = None

        return cls(model, node, window=window, days=days, hist_rolling=hist_rolling, name=name)


RollingNodeFlowRecorder.register()


#class IndexedArrayParameter(Parameter):
#    """Parameter which uses an IndexParameter to index an array of Parameters
#    An example use of this parameter is to return a demand saving factor (as
#    a float) based on the current demand saving level (calculated by an
#    `IndexParameter`).
#    Parameters
#    ----------
#    index_parameter : `IndexParameter`
#    params : iterable of `Parameters` or floats
#    Notes
#    -----
#    Float arguments `params` are converted to `ConstantParameter`
#    """

#    def __init__(self, model, index_parameter, params, **kwargs):
#        super().__init__(model, **kwargs)
#        assert(isinstance(index_parameter, IndexParameter))
#        self.index_parameter = index_parameter
#        self.children.add(index_parameter)

#        self.params = []
#        for p in params:
#            if not isinstance(p, Parameter):
#                p = ConstantParameter(model, p)
#                from pywr.parameters import ConstantParameter
#            self.params.append(p)

#        for param in self.params:
#            self.children.add(param)
#        self.children.add(index_parameter)

#    def value(self, timestep, scenario_index):
#        """Returns the value of the Parameter at the current index"""
#        #index = self.index_parameter.get_index(scenario_index)
        
#        index = self.index_parameter.get_integer_variables()[0]
#        parameter = self.params[index]
#        return parameter.get_value(scenario_index)

#    @classmethod
#    def load(cls, model, data):
#        index_parameter = load_parameter(model, data.pop("index_parameter"))
#        try:
#            parameters = data.pop("params")
#        except KeyError:
#            parameters = data.pop("parameters")
#        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
#        return cls(model, index_parameter, parameters, **data)

        
#IndexedArrayParameter.register()

# ==============================================================================
# From here some parameters created by Mikiyas for the Incomati Basin model
# ==============================================================================
class Domestic_deamnd_projection_parameter(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, Annual_increase_in_percent, **kwargs):
        super().__init__(model, **kwargs)
        self.Annual_increase_in_percent = Annual_increase_in_percent
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        
    def value(self, timestep, scenario_index):    
        
        ts_start_year = self.model.timestepper.start.year
        ts_year=self.model.timestepper.current.year
        year_diff = ts_year - ts_start_year
        projected_demand_factor = self.Annual_increase_in_percent**year_diff
        return projected_demand_factor
        
    @classmethod
    def load(cls, model, data):
        Annual_increase_in_percent = data.pop("Annual_increase_in_percent")
    
        return cls(model, Annual_increase_in_percent, **data)
    
Domestic_deamnd_projection_parameter.register()


class Demand_informed_release_Driekoppies(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, **kwargs):
        super().__init__(model, **kwargs)
        self.demand_nodes = demand_nodes
        self.DS_Komati_Lomati = DS_Komati_Lomati        
        
        self.Demand_storage_Maguga = Demand_storage_Maguga
        self.Demand_storage_Driekoppies = Demand_storage_Driekoppies 
        self.Maguga =  Maguga
        self.Driekoppies = Driekoppies
        self.Buffer_volume_Driekoppies = Buffer_volume_Driekoppies 
        self.Buffer_volume_Maguga = Buffer_volume_Maguga 

    def setup(self):
        super().setup()

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        Buffer_volume_Driekoppies = self.Buffer_volume_Driekoppies.value(ts, scenario_index) 
        Buffer_volume_Maguga = self.Buffer_volume_Maguga.value(ts, scenario_index) 
    
        Maguga_volume = self.Maguga.volume[scenario_index.global_id]
        Driekoppies_volume = self.Driekoppies.volume[scenario_index.global_id]

        if ts.index == 0:
            self.Irrigation_us_Driekoppies = ["X14_RR1","X14_RR2","X14_RR3", "X14_RR5","X14_RR6","X14_RR8"]
            Irrigation_ds_Driekoppies = ["X14_RR5","X14_RR6","X14_RR8"]
            self.Irrigation_us_Driekoppies_max_flow = {node:load_parameter(self.model, node+".Demand") for node in self.Irrigation_us_Driekoppies} 


        #if self.Driekoppies.volume[scenario_index.global_id] < Buffer_volume_Driekoppies:
        #    for node in self.Irrigation_us_Driekoppies:
        #        self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Driekoppies_max_flow[node].get_value(scenario_index)*0.5
        #else:
        #    for node in self.Irrigation_us_Driekoppies:
        #        self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Driekoppies_max_flow[node].get_value(scenario_index)

        demand = [node.get_max_flow(scenario_index) for node in self.demand_nodes]
        DS_Komati_Lomati = [node.get_max_flow(scenario_index) for node in self.DS_Komati_Lomati]

        #if both Maguga and Driekoppies volume are above the buffer storage volume 
        if (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies): 
            net_storage_Maguga = Maguga_volume - Buffer_volume_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - Buffer_volume_Driekoppies 
            Driekoppies_precent_relese = (net_storage_Driekoppies)/(net_storage_Maguga + net_storage_Driekoppies)
            Driekoppies_precent_relese = 0 if np.isnan(Driekoppies_precent_relese) else Driekoppies_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Driekoppies_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            net_storage_Maguga = Maguga_volume - self.Demand_storage_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - self.Demand_storage_Driekoppies 
            Driekoppies_precent_relese = (net_storage_Driekoppies)/(net_storage_Maguga + net_storage_Driekoppies)
            Driekoppies_precent_relese = 0 if np.isnan(Driekoppies_precent_relese) else Driekoppies_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Driekoppies_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies):
            release = sum(demand) + sum(DS_Komati_Lomati)

        elif (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            release = sum(demand) 

        return release

    @classmethod
    def load(cls, model, data):
        demand_nodes = [model._get_node_from_ref(model, node) for node in data.pop("demand_nodes")]
        DS_Komati_Lomati = [model._get_node_from_ref(model, node) for node in data.pop("DS_Komati_Lomati")]

        Maguga =  model._get_node_from_ref(model, data.pop("Maguga_Dam"))
        Driekoppies = model._get_node_from_ref(model, data.pop("Driekoppies_Dam"))
        Demand_storage_Maguga = data.pop("Demand_storage_Maguga")
        Demand_storage_Driekoppies = data.pop("Demand_storage_Driekoppies")

        Buffer_volume_Driekoppies = load_parameter(model, data.pop("Buffer_volume_Driekoppies"))
        Buffer_volume_Maguga = load_parameter(model, data.pop("Buffer_volume_Maguga"))

        return cls(model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, **data)

Demand_informed_release_Driekoppies.register()


class High_assurance_level(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, High_assurance_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.High_assurance_names = High_assurance_nodes 
        self.High_assurance_nodes = [model._get_node_from_ref(model, node) for node in self.High_assurance_names]
        
    def setup(self):
        super().setup()
        self.demand = []
        
    def probability_exceedance_plot(self, values):
        # Sort the values in ascending order
        sorted_values = np.sort(values)
        # Calculate the exceedance probabilities
        exceedance_probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        exceedance_probabilities = 1 - exceedance_probabilities
                # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values, exceedance_probabilities, marker='o', linestyle='-')
        plt.xlabel('Value')
        plt.ylabel('Exceedance Probability')
        plt.title('Probability Exceedance Plot')
        plt.grid(True)
        plt.show()

    def after(self):
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names}  
        agg_demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 
        for node in self.High_assurance_names:
            allocated_demand = self.model._get_node_from_ref(self.model, node)
            potential_demand = self.max_demand[node].get_value(self.scenario_id) 
            
            agg_demand.append(potential_demand - allocated_demand.flow[self.scenario_id.global_id])

        self.demand.append(sum(agg_demand))

        if not self.ts.year % 10 and self.ts.month == 12 and self.ts.day == 31:
            #self.probability_exceedance_plot(self.demand)
            pass

        return 0

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        self.ts = ts
        self.scenario_id = scenario_index

        agg_demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 
        for node in self.High_assurance_names:
            allocated_demand = self.model._get_node_from_ref(self.model, node)
            potential_demand = self.max_demand[node].get_value(scenario_index) 
            
            agg_demand.append(potential_demand - allocated_demand.prev_flow[scenario_index.global_id])

        """
        self.demand.append(sum(agg_demand))

        if ts.year % 3 and ts.month == 12:
            self.probability_exceedance_plot(self.demand)
        """
        return 0

    @classmethod
    def load(cls, model, data):
        High_assurance_nodes = data.pop("High_assurance_nodes")
        return cls(model, High_assurance_nodes, **data)

High_assurance_level.register()


class Low_assurance_level(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, High_assurance_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.High_assurance_names = High_assurance_nodes 
        self.High_assurance_nodes = {node: model._get_node_from_ref(model, node) for node in self.High_assurance_names}
        
    def setup(self):
        super().setup()
        self.allocated_demand = 0
        self.potential_demand = {node: 0 for node in self.High_assurance_names}
        self.demand = []
        self.max_demand = {node:load_parameter(self.model, node+".Demand") for node in self.High_assurance_names} 

    def probability_exceedance_plot(self, values):
        # Sort the values in ascending order
        sorted_values = np.sort(values)
        # Calculate the exceedance probabilities
        exceedance_probabilities = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        exceedance_probabilities = 1 - exceedance_probabilities
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values, exceedance_probabilities, marker='o', linestyle='-')
        plt.xlabel('Value')
        plt.ylabel('Exceedance Probability')
        plt.title('Probability Exceedance Plot')
        plt.grid(True)
        plt.show()

    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        self.ts = ts
        self.scenario_id = scenario_index

        agg_demand = []

        for node in self.High_assurance_names:
            self.allocated_demand = self.model._get_node_from_ref(self.model, node)
            agg_demand.append(self.potential_demand[node] - self.allocated_demand.prev_flow[scenario_index.global_id])
            self.potential_demand[node] = self.max_demand[node].get_value(scenario_index) 

        if not ts.index == 0:
            self.demand.append(sum(agg_demand))


        if not self.ts.year % 5 and self.ts.month == 12 and self.ts.day == 31:
            #self.probability_exceedance_plot(self.demand)
            pass
            
        return 0

    @classmethod
    def load(cls, model, data):
        High_assurance_nodes = data.pop("High_assurance_nodes")
        return cls(model, High_assurance_nodes, **data)

Low_assurance_level.register()


class Simple_Irr_demand_calculator_with_file(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, evaporation,rainfall_factor,IRR_exp,IRR_eff,crop_ET,crop_area,Max_area,rainfall,index_col, **kwargs):
        super().__init__(model, **kwargs)
        self.evaporation = [x/30 for x in evaporation]
        self.rainfall_factor = rainfall_factor
        self.IRR_exp = IRR_exp
        self.IRR_eff = IRR_eff
        self.crop_ET = crop_ET
        self.crop_area = crop_area
        self.Max_area = Max_area
        self.rainfall_ = rainfall
        self.index_col=index_col
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self.rainfall_=pd.read_hdf(self.rainfall_)
        
    def value(self, timestep, scenario_index):    
        
        ts = self.model.timestepper.current
        month=ts.month
        year=ts.year
        self.rainfall=self.rainfall_[self.index_col][str(year)+"_"+str(month)]/ts.day
        
        rainfall_factor=self.rainfall_factor
                
        evaporation=self.evaporation[month-1]
        Total_Irr_demand=0
        for x in self.crop_ET.keys():
            if x in self.crop_area.keys():
                Net_demand=max((self.crop_ET[x][month-1]*evaporation-rainfall_factor[month-1]*self.rainfall),0)
                Irr_eff=1+(1-0.80)
                Irr_demand=Net_demand*(self.crop_area[x]*0.01*1.05)*self.Max_area*Irr_eff * 1e6 * 1e-3 * 1e-6
                
                Total_Irr_demand+=Irr_demand
        return Total_Irr_demand
        
    @classmethod
    def load(cls, model, data):
        evaporation = data.pop("evaporation")
        rainfall_factor = data.pop("rainfall_factor")
        IRR_exp = data.pop("IRR_exp")
        IRR_eff = data.pop("IRR_eff")
        crop_ET = data.pop("crop_ET")
        crop_area = data.pop("crop_area")
        Max_area = data.pop("Max_area")
        rainfall = data.pop("rainfall")
        index_col=data.pop("index_col")
        
        return cls(model, evaporation,rainfall_factor,IRR_exp,IRR_eff,crop_ET,crop_area,Max_area,rainfall, index_col,**data)

Simple_Irr_demand_calculator_with_file.register()


class Demand_informed_release_Maguga(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, Maguga_hydropower_release_m3_s, Maguga_hydropower_operation_hours_per_week, **kwargs):
        super().__init__(model, **kwargs)
        self.demand_nodes = demand_nodes
        self.DS_Komati_Lomati = DS_Komati_Lomati
        self.Demand_storage_Maguga = Demand_storage_Maguga
        self.Demand_storage_Driekoppies = Demand_storage_Driekoppies 
        self.Maguga =  Maguga
        self.Driekoppies = Driekoppies
        self.Buffer_volume_Driekoppies = Buffer_volume_Driekoppies 
        self.Buffer_volume_Maguga = Buffer_volume_Maguga 

        self.Maguga_hydropower_release_m3_s = Maguga_hydropower_release_m3_s
        self.Maguga_hydropower_operation_hours_per_week = Maguga_hydropower_operation_hours_per_week 

    def setup(self):
        super().setup()
        self.Irrigation_us_Maguga = ["X11_RR3","X11_RR4","X11_RR16","X11_RR7","X11_RR9","X11_RR11","X11_RR13","X12_RR12","X12_RR13","X12_RR9","X12_RR7","X13_RR1","X13_RR3","X13_RR5","X13_RR7","X13_RR12","X13_RR2"]
        self.Irrigation_us_Maguga_max_flow = {node:load_parameter(self.model, node+".Demand") for node in self.Irrigation_us_Maguga} 
        self.peak_hours_relese = self.Maguga_hydropower_release_m3_s["peak_hours_relese"] 
        self.standard_hours_generation = self.Maguga_hydropower_release_m3_s["standard_hours_generation"] 
        self.off_peak_hours_relese = self.Maguga_hydropower_release_m3_s["off_peak_hours_relese"] 

        self.peak_hours = load_parameter(self.model, self.Maguga_hydropower_operation_hours_per_week["peak_hours"])
        self.standard_hours = load_parameter(self.model, self.Maguga_hydropower_operation_hours_per_week["standard_hours"])
        
    def value(self, ts, scenario_index):
        
        ts = self.model.timestepper.current
        
        self.peak_hour = self.peak_hours.get_value(scenario_index)
        self.standard_hour = self.standard_hours.get_value(scenario_index)
        self.off_peak_hour = 168 - self.peak_hour - self.standard_hour

        factor = 3600/7e6
        self.hydro_release = (self.peak_hour*self.peak_hours_relese + self.standard_hour*self.standard_hours_generation + self.off_peak_hour*self.off_peak_hours_relese)*factor


        Buffer_volume_Driekoppies = self.Buffer_volume_Driekoppies.value(ts, scenario_index) 
        Buffer_volume_Maguga = self.Buffer_volume_Maguga.value(ts, scenario_index) 
        Maguga_volume = self.Maguga.volume[scenario_index.global_id]
        Driekoppies_volume = self.Driekoppies.volume[scenario_index.global_id]


        if self.Maguga.volume[scenario_index.global_id] < Buffer_volume_Maguga:
            for node in self.Irrigation_us_Maguga:
                self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Maguga_max_flow[node].get_value(scenario_index)
        else:
            for node in self.Irrigation_us_Maguga:
                self.model._get_node_from_ref(self.model, node).max_flow = self.Irrigation_us_Maguga_max_flow[node].get_value(scenario_index)

        #Todo
        # 1. identfy the hydropwoer requirment as an input parameter. this could be the montly required release based on the Peak, Standard and __
        # 2. compaire the downstream release with the hydropower demand and if the hydropower release is above the demand, adjust the release 
        # 3. make sure that the demand is satisfied unless there is no water in the reservoir.
        # 4. make sure that the release form the reservoir should consider the spill from the reservoir


        demand = [node.get_max_flow(scenario_index) for node in self.demand_nodes]
        DS_Komati_Lomati = [node.get_max_flow(scenario_index) for node in self.DS_Komati_Lomati]
        #if both Maguga and Driekoppies volume are above the buffer storage volume
        Maguga_precent_relese = 0 
        if (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies): 
            net_storage_maguga = Maguga_volume - Buffer_volume_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - Buffer_volume_Driekoppies 
            Maguga_precent_relese = (net_storage_maguga)/(net_storage_maguga + net_storage_Driekoppies)
            Maguga_precent_relese = 0 if np.isnan(Maguga_precent_relese) else Maguga_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Maguga_precent_relese

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            net_storage_maguga = Maguga_volume - self.Demand_storage_Maguga 
            net_storage_Driekoppies = Driekoppies_volume - self.Demand_storage_Driekoppies 
            Maguga_precent_relese = (net_storage_maguga)/(net_storage_maguga + net_storage_Driekoppies)
            Maguga_precent_relese = 0 if np.isnan(Maguga_precent_relese) else Maguga_precent_relese
            release = sum(demand) + sum(DS_Komati_Lomati)*Maguga_precent_relese

        elif (Maguga_volume > Buffer_volume_Maguga and Driekoppies_volume < Buffer_volume_Driekoppies):
            release = sum(demand) + sum(DS_Komati_Lomati)

        elif (Maguga_volume < Buffer_volume_Maguga and Driekoppies_volume > Buffer_volume_Driekoppies):
            release = sum(demand) 
     
        # release for hydropower  
        if release < self.hydro_release:
            release = self.hydro_release

        return release

    @classmethod
    def load(cls, model, data):
        demand_nodes = [model._get_node_from_ref(model, node) for node in data.pop("demand_nodes")]
        DS_Komati_Lomati = [model._get_node_from_ref(model, node) for node in data.pop("DS_Komati_Lomati")]
        Maguga =  model._get_node_from_ref(model, data.pop("Maguga_Dam"))
        Driekoppies = model._get_node_from_ref(model, data.pop("Driekoppies_Dam"))
        Demand_storage_Maguga = data.pop("Demand_storage_Maguga")
        Demand_storage_Driekoppies = data.pop("Demand_storage_Driekoppies")

        Buffer_volume_Driekoppies = load_parameter(model, data.pop("Buffer_volume_Driekoppies"))
        Buffer_volume_Maguga = load_parameter(model, data.pop("Buffer_volume_Maguga"))

        Maguga_hydropower_release_m3_s = data.pop("Maguga_hydropower_release_m3_s")
        Maguga_hydropower_operation_hours_per_week = data.pop("Maguga_hydropower_operation_hours_per_week")

        return cls(model, demand_nodes, DS_Komati_Lomati, Maguga, Driekoppies, Buffer_volume_Driekoppies, Buffer_volume_Maguga, Demand_storage_Maguga, Demand_storage_Driekoppies, Maguga_hydropower_release_m3_s, Maguga_hydropower_operation_hours_per_week, **data)

Demand_informed_release_Maguga.register()