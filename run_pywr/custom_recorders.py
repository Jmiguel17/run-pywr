import os
import sys
import numpy as np
import pandas as pd

from pywr.parameters import load_parameter, Parameter
from pywr.recorders import (NumpyArrayNodeRecorder, NodeRecorder, Aggregator, NumpyArrayStorageRecorder, NumpyArrayAbstractStorageRecorder, 
                            Recorder, hydropower_calculation, NumpyArrayParameterRecorder, BaseConstantParameterRecorder)

class NumpyArrayAnnualNodeDeficitFrequencyRecorder(NodeRecorder):

    """
    Number of two consecutive years where there is a deficit greater that a threshold
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'COUNT_NONZERO')
        
        super().__init__(model, node, **kwargs)
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.threshold = threshold

        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)

            if node.flow[scenario_index.global_id] < max_flow * self.threshold:
                
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def to_dataframe(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(np.array(self._data), index=index, columns=sc_index)


    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        count_nonzeros = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').sum().to_numpy()

        return self._temporal_aggregator.aggregate_2d(count_nonzeros, axis=0, ignore_nan=self.ignore_nan)

    def aggregated_value(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_val = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').sum().to_numpy()

        zeros_ones = np.where(annual_val > 0, 1, 0)

        tuples = []
        for i in range(0, len(zeros_ones), 1):
            if i == 0:
                continue
            else:
                tem = [zeros_ones[i-1], zeros_ones[i]]
                tuples.append(tem)

        count = sum([all(np.array(x) == 1) for x in tuples])

        return count


NumpyArrayAnnualNodeDeficitFrequencyRecorder.register()


class AbstractComparisonNodeRecorder(NumpyArrayNodeRecorder):
    """ Base class for all Recorders performing timeseries comparison of `Node` flows
    """

    def __init__(self, model, node, observed, obs_freq=None, **kwargs):
        super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs)

        self.observed = observed
        self._aligned_observed = None
        self.obs_freq = obs_freq

    def setup(self):
        super(AbstractComparisonNodeRecorder, self).setup()
        # Align the observed data to the model

        from pywr.parameters import align_and_resample_dataframe

        freq = self.obs_freq
        index_col = self.observed.index.tolist()
        start, end = index_col[0], index_col[-1]
        timestepper = pd.period_range(start=start, end=end, freq=freq)
        self._aligned_observed = align_and_resample_dataframe(self.observed, timestepper, 'sum')
        #self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

    @classmethod
    def load(cls, model, data):
        # called when the parameter is loaded from a JSON document

        observed = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq")

        if '.csv' in url:
            data_observed = pd.read_csv(url)

        if 'xlsx' in url:
            data_observed = pd.read_excel(url)

        observed = pd.DataFrame(data=np.array(data_observed[observed]), index=data_observed[index_col])

        node = model._get_node_from_ref(model, data.pop("node"))

        return cls(model, node, observed, obs_freq, **data)


class RootMeanSquaredErrorNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the RMSE between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs, mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        val = np.sqrt(np.mean((obs - mod) ** 2, axis=0))

        return val.values


RootMeanSquaredErrorNodeRecorder.register()


class NashSutcliffeEfficiencyNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the Nash-Sutcliffe efficiency model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed

        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs,mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        obs_mean = np.mean(obs, axis=0)

        val = 1.0 - np.sum((obs-mod)**2, axis=0)/np.sum((obs-obs_mean)**2, axis=0)

        return val.values


NashSutcliffeEfficiencyNodeRecorder.register()


class PercentBiasNodeRecorder(AbstractComparisonNodeRecorder):
    """ Recorder evaluates the percent bias between model and observed """
    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed
        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model.')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

            #mod = pandas.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).sum()

        new = pd.merge(obs,mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        val = np.sum(obs-mod, axis=0)*100/np.sum(obs, axis=0)

        return val.values


PercentBiasNodeRecorder.register()


class AbstractComparisonStorageRecorder(NumpyArrayStorageRecorder):
    """ Base class for all Recorders performing timeseries comparison of `Storage Node`
    """

    def __init__(self, model, node, observed, obs_freq=None, **kwargs):
        super(AbstractComparisonStorageRecorder, self).__init__(model, node, **kwargs)

        self.observed = observed
        self._aligned_observed = None
        self.obs_freq = obs_freq

    def setup(self):
        super(AbstractComparisonStorageRecorder, self).setup()
        # Align the observed data to the model

        from pywr.parameters import align_and_resample_dataframe
        #        self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

        freq = self.obs_freq
        index_col = self.observed.index.tolist()
        start, end = index_col[0], index_col[-1]
        timestepper = pd.period_range(start=start, end=end, freq=freq)
        self._aligned_observed = align_and_resample_dataframe(self.observed, timestepper, 'mean')

    @classmethod
    def load(cls, model, data):
        # called when the parameter is loaded from a JSON document

        observed = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq")

        if '.csv' in url:
            data_observed = pd.read_csv(url)

        if 'xlsx' in url:
            data_observed = pd.read_excel(url)

        observed = pd.DataFrame(data=np.array(data_observed[observed]), index=data_observed[index_col])

        node = model._get_node_from_ref(model, data.pop("node"))

        return cls(model, node, observed, obs_freq, **data)


class NashSutcliffeEfficiencyStorageRecorder(AbstractComparisonStorageRecorder):
    """ Recorder evaluates the Nash-Sutcliffe efficiency model and observed """

    def values(self):

        freq = self.obs_freq
        obs = self._aligned_observed

        mod = self.data

        if freq is None:
            mod = self.data
        else:
            if self.model.timestepper.freq == freq:
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index).resample(freq).mean()
            else:
                #print(f'OJO! The recorder associated to this node "{self.node.name}" '
                      #f'has freq observed data =! freq model - '
                      #f'Check if freq observed data >= freq model')
                mod = pd.DataFrame(self.data, index=self.model.timestepper.datetime_index.astype('datetime64[ns]')).resample(freq).mean()
                mod.index = mod.index.strftime('%Y-%m')
                obs.index = obs.index.astype('datetime64[ns]').strftime('%Y-%m')

        new = pd.merge(obs, mod, how='inner', left_index=True, right_index=True)
        obs = new.iloc[:, 0].to_frame().T.reset_index(drop=True).T
        mod = new.iloc[:, 1].to_frame().T.reset_index(drop=True).T

        obs_mean = np.mean(obs, axis=0)

        val = 1.0 - np.sum((obs - mod) ** 2, axis=0) / np.sum((obs - obs_mean) ** 2, axis=0)

        return val.values


NashSutcliffeEfficiencyStorageRecorder.register()


class ReservoirMonthlyReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total months below minimum storage level / total months in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):

        raise NotImplementedError()


ReservoirMonthlyReliabilityRecorder.register()


class ReservoirAnnualReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total years below minimum storage level / total years in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirAnnualReliabilityRecorder.register()


class SupplyReliabilityRecorder(NodeRecorder):

    """
    add description
    """

    def __init__(self, model, node, **kwargs):
        super().__init__(model, node, **kwargs)

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)

            if max_flow == 0:
                deficit = 0

            else:
                deficit  = (max_flow - node.flow[scenario_index.global_id]) / max_flow

            if deficit > 0.01:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max().loc[:str(last_year), :]

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


SupplyReliabilityRecorder.register()


class AnnualDeficitRecorder(NodeRecorder):

    """
    Annual deficit recorder (%)
    """

    def __init__(self, model, node, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        rlts = 1 - supply.divide(demand)

        if self.temporal_aggregator == 'mean':
            to_save = rlts.mean()

        if self.temporal_aggregator == 'max':
            to_save = rlts.max()

        if self.temporal_aggregator == 'min':
            to_save = rlts.min()

        return to_save
    
    def to_dataframe(self):
        
        raise NotImplementedError()


AnnualDeficitRecorder.register()


class ReservoirResilienceRecorder(NumpyArrayAbstractStorageRecorder):

    """
    add description
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        tem_dams = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        tem_dams_diff = tem_dams.diff().ne(0).cumsum()
        
        tem_dams_occurrence = tem_dams.multiply(tem_dams_diff)

        resilience = {}
        
        levels = [x for x, _ in enumerate(tem_dams_occurrence.columns.names)]
        
        for idx, dataframe in tem_dams_occurrence.groupby(level=levels, axis=1):
            
            tem = dataframe.T.reset_index(drop=True).T
            
            tem.columns = ['col']
            
            tem_res = tem[tem['col'] != 0].groupby(['col'])['col'].count()

            if self.temporal_aggregator == 'mean':
                resilience[idx] = tem_res.mean()
                
            if self.temporal_aggregator == 'max':
                resilience[idx] = tem_res.max()
            
        
        rlts = pd.DataFrame.from_dict(resilience, orient='index', columns=[""])

        rlts = rlts.T

        rlts.columns = pd.MultiIndex.from_tuples(rlts.columns, names=sc_index.names)

        return rlts.T
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirResilienceRecorder.register()


class RelativeCropYieldRecorder(Recorder):
    """Relative crop yield recorder.

    This recorder computes the relative crop yield based on a curtailment ratio between a node's
    actual flow and it's `max_flow` expected flow. It is assumed the `max_flow` parameter is an
    `AggregatedParameter` containing only `IrrigationWaterRequirementParameter` parameters.

    """
    def __init__(self, model, nodes, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)

        for node in nodes:
            max_flow_param = node.max_flow
            self.children.add(max_flow_param)

        self.nodes = nodes
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.data = None

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self.data = np.zeros((nts, ncomb))

    def reset(self):
        self.data[:, :] = 0.0

    def after(self):

        norm_crop_revenue = None
        full_norm_crop_revenue = None
        ts = self.model.timestepper.current
        self.data[ts.index, :] = 0
        norm_yield = 0
        full_norm_yield = 0

        for node in self.nodes:
            crop_aggregated_parameter = node.max_flow
            actual = node.flow
            requirement = np.array(crop_aggregated_parameter.get_all_values())
            # Divide non-zero elements
            curtailment_ratio = np.divide(actual, requirement, out=np.zeros_like(actual), where=requirement != 0)
            no_curtailment = np.ones_like(curtailment_ratio)

            if norm_crop_revenue is None:
                norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(curtailment_ratio)
                full_norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(no_curtailment)

            for parameter in crop_aggregated_parameter.parameters:
                crop_revenue = parameter.crop_revenue(curtailment_ratio)
                full_crop_revenue = parameter.crop_revenue(no_curtailment)
                crop_yield = parameter.crop_yield(curtailment_ratio)
                full_crop_yield = parameter.crop_yield(no_curtailment)
                # Increment effective yield, scaled by the first crop's revenue
                norm_yield += crop_yield * np.divide(crop_revenue, norm_crop_revenue,
                                                    out=np.zeros_like(crop_revenue),
                                                    where=norm_crop_revenue != 0)

                full_norm_yield += full_crop_yield * np.divide(full_crop_revenue, full_norm_crop_revenue,
                                                              out=np.ones_like(full_crop_revenue),
                                                              where=full_norm_crop_revenue != 0)
                
                if requirement<0.00001:
                    self.data[ts.index, :] = 99999
                else:
                    self.data[ts.index, :] = norm_yield / full_norm_yield

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self.data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self.data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        nodes = [model._get_node_from_ref(model, n) for n in data.pop('nodes')]
        return cls(model, nodes, **data)

RelativeCropYieldRecorder.register()


class AverageAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the average annual crop yield for each scenario based on the curtailment ratio between a node's
    """

    def __init__(self, model, node, threshold=None, **kwargs):

        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.threshold = threshold
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)
    
            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):
    
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        
        return curtailment_ratio.multiply(areas).multiply(yields)


    def values(self):
        
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)

        return crop_yield.mean(axis=0) #self._temporal_aggregator.aggregate_2d(crop_yield.values, axis=0, ignore_nan=self.ignore_nan) 


AverageAnnualCropYieldScenarioRecorder.register()


class TotalAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the Total annual crop yield for each scenario assuming there is anough water to irrigate the crop
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        #supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = demand.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)

        return crop_yield.mean(axis=0)


TotalAnnualCropYieldScenarioRecorder.register()


class IrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the supply reliability of an irrigation node considering only the months with higher demand 
    based on the Kc parameter. 
    
    A month with high demand > 0.8 Kc
    A year is considered that fails if the supply is less than 80% of the demand in any of the months with higher demand
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._kc = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._kc[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        max_flow_param = self.node.max_flow

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)
            self._kc[ts.index, scenario_index.global_id] = max_flow_param.crop_water_factor_parameter.get_value(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]

        # Here we calculate the months where the demand is higher than 0.8 Kc
        mths_kc = np.where(self._kc < np.max(self._kc)*0.8, 0, 1)
        mths_kc = pd.DataFrame(mths_kc, index=index, columns=sc_index).resample('M').mean().loc[:str(last_year), :]

        moths_failures = np.where(supply < demand*self.threshold, 1, 0)
        moths_failures = pd.DataFrame(moths_failures, index=demand.index, columns=demand.columns)

        # Here we calculate the years where there is a failure only considering the months with high demand "mths_kc"
        failures = moths_failures.multiply(mths_kc).dropna().resample('Y').max()


        return 1 - (failures.sum().round(0) / failures.shape[0])


IrrigationSupplyReliabilityScenarioRecorder.register()


class CropCurtailmentRatioScenarioRecorder(NodeRecorder):

    """
    This recorder save the Annual Curtailment Ratios
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        
        return curtailment_ratio


    def values(self):
        

        return NotImplementedError()


CropCurtailmentRatioScenarioRecorder.register()


class AnnualIrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the annual supply reliability based on a threashold. 

    A year is considered that fails if the supply is less than a threashold
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]


        # Here we calculate the years where there is a failure only considering the threshold
        failures = np.where(supply < demand*self.threshold, 1, 0)
        failures = pd.DataFrame(failures, index=demand.index, columns=demand.columns)

        return 1 - (failures.sum().round(0) / failures.shape[0])


AnnualIrrigationSupplyReliabilityScenarioRecorder.register()


class AverageAnnualIrrigationRevenueScenarioRecorder(NodeRecorder):

    """
    This recorder computes the average annual irrigation revenue for each scenario based on the curtailment ratio between a node's
    the price should be imput in $/tn
    """

    def __init__(self, model, node, threshold=None, price=1, **kwargs):

        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.threshold = threshold
        self.price = price
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._yield = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._yield[:, :] = 0.0
        self._area[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

            if isinstance(node.max_flow.yield_per_area, float):
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area
            else:
                self._yield[ts.index, scenario_index.global_id] = node.max_flow.yield_per_area.get_value(scenario_index)

            if isinstance(node.max_flow.area, float):
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area
            else:
                self._area[ts.index, scenario_index.global_id] = node.max_flow.area.get_value(scenario_index)

        return 0

    def to_dataframe(self):

        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)
        
        return crop_yield.divide(1e3).multiply(self.price).divide(1e6) # kg to tn then $ to M$


    def values(self):

        max_flow_param = self.node.max_flow
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        areas = pd.DataFrame(np.array(self._area), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]
        yields = pd.DataFrame(np.array(self._yield), index=index, columns=sc_index).resample('Y').mean().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0
        curtailment_ratio.fillna(0, inplace=True)

        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(yields, axis=1)
        revenue = crop_yield.divide(1e3).multiply(self.price).divide(1e6) # kg to tn then $ to M$

        return self._temporal_aggregator.aggregate_2d(revenue.dropna().values, axis=0, ignore_nan=self.ignore_nan)
    

AverageAnnualIrrigationRevenueScenarioRecorder.register()
    
                
class AnnualSeasonalAccumulatedFlowRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, months=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.months = months
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self.cummulatedFlow = np.zeros((nts, ncomb))

    def reset(self):
        self.cummulatedFlow[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self.cummulatedFlow[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualFlow = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualFlow = AnnualFlow.resample('D').ffill()

        AnnualFlow = AnnualFlow[AnnualFlow.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualFlow = AnnualFlow.loc[:str(last_year), :].resample('Y').sum()
        
        return AnnualFlow


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualFlow = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualFlow = AnnualFlow.resample('D').ffill()
        AnnualFlow = AnnualFlow[AnnualFlow.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualFlow = AnnualFlow.loc[:str(last_year), :].resample('Y').sum()

        return self._temporal_aggregator.aggregate_2d(AnnualFlow.values, axis=0, ignore_nan=self.ignore_nan)


AnnualSeasonalAccumulatedFlowRecorder.register()


class AnnualSeasonalVolumeRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, months=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.months = months
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self.cummulatedFlow = np.zeros((nts, ncomb))

    def reset(self):
        self.cummulatedFlow[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self.cummulatedFlow[ts.index, scenario_index.global_id] = node.volume[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualVolume = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualVolume = AnnualVolume.resample('D').ffill()
        AnnualVolume = AnnualVolume[AnnualVolume.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualVolume = AnnualVolume.loc[:str(last_year), :].resample('Y').mean()
        
        return AnnualVolume


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        AnnualVolume = pd.DataFrame(np.array(self.cummulatedFlow), index=index, columns=sc_index)
        AnnualVolume = AnnualVolume.resample('D').ffill()
        AnnualVolume = AnnualVolume[AnnualVolume.index.month.isin(self.months)]

        last_year = index[-1].year
        AnnualVolume = AnnualVolume.loc[:str(last_year), :].resample('Y').mean()
        
        #AnnualVolume = AnnualVolume.mean(axis=0)
        
        # get the value only
        #AnnualVolume = AnnualVolume.values

        return self._temporal_aggregator.aggregate_2d(AnnualVolume.values, axis=0, ignore_nan=self.ignore_nan) 


AnnualSeasonalVolumeRecorder.register()

class AnnualHydropowerRecorder(NumpyArrayNodeRecorder):
    """ Calculates the annual power production using the hydropower equation
    This recorder is inspired on the `pywr.recorders.HydropowerRecorder` but it is
    designed to be used with the `pywr.recorders.NumpyArrayNodeRecorder` class. 

    Parameters
    ----------

    water_elevation_parameter : Parameter instance (default=None)
        Elevation of water entering the turbine. The difference of this value with the `turbine_elevation` gives
        the working head of the turbine.
    turbine_elevation : double
        Elevation of the turbine itself. The difference between the `water_elevation` and this value gives
        the working head of the turbine.
    efficiency : float (default=1.0)
        The efficiency of the turbine.
    density : float (default=1000.0)
        The density of water.
    flow_unit_conversion : float (default=1.0)
        A factor used to transform the units of flow to be compatible with the equation here. This
        should convert flow to units of :math:`m^3/day`
    energy_unit_conversion : float (default=1e-6)
        A factor used to transform the units of total energy. Defaults to 1e-6 to return :math:`MJ`.

    Notes
    -----
    The hydropower calculation uses the following equation.

    .. math:: P = \\rho * g * \\delta H * q

    The flow rate in should be converted to units of :math:`m^3` per day using the `flow_unit_conversion` parameter.

    Head is calculated from the given `water_elevation_parameter` and `turbine_elevation` value. If water elevation
    is given then head is the difference in elevation between the water and the turbine. If water elevation parameter
    is `None` then the head is simply the turbine elevation.


    See Also
    --------
    TotalHydroEnergyRecorder
    pywr.parameters.HydropowerTargetParameter

    """
    def __init__(self, model, node, monthly_seasonality, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs) # Changed from super(HydropowerRecorder, self) for Python 3+ style

        # Initialize _water_elevation_parameter before setting the property
        # to ensure the setter can access it if it checks hasattr(self, '_water_elevation_parameter').
        # However, direct assignment to the property will call the setter.
        self._water_elevation_parameter = None 
        self.water_elevation_parameter = water_elevation_parameter # Use the setter

        self._monthly_seasonality = monthly_seasonality
        self.turbine_elevation = float(turbine_elevation)
        self.efficiency = float(efficiency)
        self.density = float(density)
        self.flow_unit_conversion = float(flow_unit_conversion)
        self.energy_unit_conversion = float(energy_unit_conversion)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    @property
    def water_elevation_parameter(self):
        """The water elevation parameter instance."""
        return self._water_elevation_parameter

    @water_elevation_parameter.setter
    def water_elevation_parameter(self, parameter):
        """Sets the water elevation parameter, updating children."""
        current_parameter = getattr(self, '_water_elevation_parameter', None)

        if current_parameter: # If current_parameter is not None and truthy
            if current_parameter in self.children:
                self.children.remove(current_parameter)
        
        # The original Cython code `self.children.add(parameter)` would add the parameter
        # to the set `self.children` even if `parameter` is None.
        # This behavior is preserved here.
        self.children.add(parameter)
        
        self._water_elevation_parameter = parameter

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0

    def after(self):
        """Called after each timestep to record hydropower production."""
        # Type hints for clarity, not strict enforcement in standard Python
        # q: float
        # head: float
        # power: float
        
        ts = self.model.timestepper.current
        # scenario_index: ScenarioIndex # Type hint for loop variable

        # Assuming self.node is set by the parent class (NumpyArrayNodeRecorder or NodeRecorder)
        # If not, and self._node is used:
        # node_flow_attr = self._node.flow # Or however flow is accessed per scenario

        for scenario_index in self.model.scenarios.combinations:
            
            if self._water_elevation_parameter is not None:
                water_elev = self._water_elevation_parameter.get_value(scenario_index)
                head = water_elev - self.turbine_elevation
            else:
                # If water_elevation_parameter is None, head is taken as turbine_elevation.
                # This matches the docstring and the simplified logic from the Cython code
                # given turbine_elevation is always a float.
                head = self.turbine_elevation
            
            # Negative head is not physically valid for power generation
            head = max(head, 0.0)

            # Get the flow from the current node for the specific scenario
            # NodeRecorder (parent of NumpyArrayNodeRecorder) stores the node as self._node.
            # Flow for a scenario is typically accessed like this in Pywr:
            q = self.node.flow[scenario_index.global_id]

            # Calculate power using the external hydropower_calculation function
            power = hydropower_calculation(
                q, 
                head, 
                0.0,  # Assuming 0.0 for tailwater_elevation as in the Cython call
                self.efficiency, 
                density=self.density,
                flow_unit_conversion=self.flow_unit_conversion,
                energy_unit_conversion=self.energy_unit_conversion
            )
            
            # Store the calculated power
            # self._data is assumed to be a 2D NumPy array (timesteps, scenarios)
            self._data[ts.index, scenario_index.global_id] = power

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_hydropower = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        annual_hydropower = annual_hydropower.resample('D').ffill()

        if self._monthly_seasonality is not None:
            annual_hydropower = annual_hydropower[annual_hydropower.index.month.isin(self._monthly_seasonality)]

        annual_hydropower = annual_hydropower.resample('Y').sum() # To get annual hydropower generation in MWh/year

        if self.factor is not None:
            annual_hydropower = annual_hydropower.multiply(self.factor, axis=0)
    
        return self._temporal_aggregator.aggregate_2d(annual_hydropower.values, axis=0, ignore_nan=self.ignore_nan) # Return revenues if a factor (price) is set
    

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        to_dataframe() is a method that returns the hydropower generation in energy units at annual scale.
        
        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        annual_hydropower = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        annual_hydropower = annual_hydropower.resample('D').ffill()

        if self._monthly_seasonality is not None:
            annual_hydropower = annual_hydropower[annual_hydropower.index.month.isin(self._monthly_seasonality)]

        return annual_hydropower.resample('Y').sum() # To get annual hydropower generation in MWh/year

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""
        # It's good practice to ensure 'pywr.parameters' is accessible
        # or handle potential ImportError if this code is part of a larger system.
        from pywr.parameters import load_parameter # Assuming pywr is structured this way

        node_name = data.pop("node")
        monthly_seasonality = data.pop("monthly_seasonality", None)
        node = model.nodes[node_name] # Get the actual node instance

        water_elevation_param_data = data.pop("water_elevation_parameter", None)
        water_elevation_parameter = None
        if water_elevation_param_data is not None:
            water_elevation_parameter = load_parameter(model, water_elevation_param_data)
        
        # Remaining items in data are passed as kwargs to __init__
        return cls(model, node, monthly_seasonality, water_elevation_parameter=water_elevation_parameter, **data)
    
    
AnnualHydropowerRecorder.register()

class SeasonalTransferConstraintRecorder(NodeRecorder):

    """
    This recorder calculates the total annual flow using the months defined per user. 
    It counts from the 1st of the first month to the last day of the last month.
    """

    def __init__(self, model, node, node_rule, monthly_seasonality=None, **kwargs):
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')

        super().__init__(model, node, **kwargs)
        self.node_rule = node_rule
        self.monthly_seasonality = monthly_seasonality
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))
        self._data_rule = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
        self._data_rule[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        node_rule = self.node_rule

        for scenario_index in self.model.scenarios.combinations:

            self._data[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._data_rule[ts.index, scenario_index.global_id] = node_rule.flow[scenario_index.global_id]

        return 0

    def to_dataframe(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        outflow = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)
        outflow = outflow.resample('D').ffill()

        outflow = outflow[outflow.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        outflow = outflow.loc[:str(last_year), :].resample('Y').sum()

        rule = pd.DataFrame(np.array(self._data_rule), index=index, columns=sc_index)
        rule = rule.resample('D').ffill()

        rule = rule[rule.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        rule = rule.loc[:str(last_year), :].resample('Y').sum()

        return (rule - 4000) - outflow


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex
        
        outflow = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)
        outflow = outflow.resample('D').ffill()

        outflow = outflow[outflow.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        outflow = outflow.loc[:str(last_year), :].resample('Y').sum()

        rule = pd.DataFrame(np.array(self._data_rule), index=index, columns=sc_index)
        rule = rule.resample('D').ffill()

        rule = rule[rule.index.month.isin(self.monthly_seasonality)]

        last_year = index[-1].year
        rule = rule.loc[:str(last_year), :].resample('Y').sum()

        constraint = (rule - 4000) - outflow

        return self._temporal_aggregator.aggregate_2d(constraint.values, axis=0, ignore_nan=self.ignore_nan)
    

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""
        
        node_name = data.pop("node")
        node_rule_name = data.pop("node_rule", None)

        node = model.nodes[node_name] # Get the actual node instance
        node_rule = model.nodes[node_rule_name] # Get the actual node instance

        monthly_seasonality = data.pop("monthly_seasonality", None)
        
        # Remaining items in data are passed as kwargs to __init__
        return cls(model, node, node_rule, monthly_seasonality, **data)


SeasonalTransferConstraintRecorder.register()


class ConveyanceEfficiencyCostRecorder(BaseConstantParameterRecorder):
    """Recorder to calculate the cost of improving canal conveyance efficiency.

    This recorder calculates the total capital cost of improving canal conveyance
    efficiency for a given length of canal. The cost is calculated using a
    power-law relationship between the efficiency gain and the investment cost.

    Attributes
    ----------
    parameter : pywr.parameters.Parameter
        A parameter that provides the target conveyance efficiency as a fraction (0-1).
    a : float
        The cost coefficient.
    b : float
        The exponent of the cost function.
    n0 : float
        The baseline conveyance efficiency (fraction).
    canal_length_km : float
        The total length of the canal in kilometers.
    """

    def __init__(self, model, parameter, a, b, n0, canal_length_km, **kwargs):

        agg_func = kwargs.pop('agg_func', 'mean')

        super().__init__(model, parameter, **kwargs)
        self.a = a
        self.b = b
        self.n0 = n0
        self.canal_length_km = canal_length_km

        self._scenario_aggregator = Aggregator(agg_func)
        self._scenario_aggregator.func = agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        self.total_cost = np.zeros((ncomb))

    def reset(self):
        self.total_cost[...] = 0.0
        
    def after(self):

        for scenario_index in self.model.scenarios.combinations:
            efficiency = 1 - self._param.get_value(scenario_index)
            delta_efficiency = np.maximum(0.0, efficiency - (1-self.n0))
            cost_per_km = self.a * (delta_efficiency ** self.b)
            
            self.total_cost[scenario_index.global_id] = cost_per_km * self.canal_length_km

        return 0

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`."""
        return self.total_cost
    
    def aggregated_value(self):
        return self._scenario_aggregator.aggregate_1d(self.total_cost, ignore_nan=self.ignore_nan)


ConveyanceEfficiencyCostRecorder.register()


class ReservoirStorageExpansionCostRecorder(BaseConstantParameterRecorder):
    """Recorder to calculate the cost of new reservoir storage.

    This recorder calculates the capital cost of building new reservoir storage
    capacity. The cost is calculated using a power function that captures
    economies of scale.

    Attributes
    ----------
    parameter : pywr.parameters.Parameter
        A parameter that provides the additional storage capacity in MCM.
    a : float
        The coefficient of the cost function.
    b : float
        The exponent of the cost function.
    """

    def __init__(self, model, parameter, current_max_volumne, a, b, **kwargs):

        agg_func = kwargs.pop('agg_func', 'mean')
        super().__init__(model, parameter, **kwargs)
        self.a = a
        self.b = b

        self.current_max_volumne = current_max_volumne

        self._scenario_aggregator = Aggregator(agg_func)
        self._scenario_aggregator.func = agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        self.total_cost = np.zeros((ncomb))

    def reset(self):
        self.total_cost[...] = 0.0

    def after(self):

        for scenario_index in self.model.scenarios.combinations:
            storage = self._param.get_value(scenario_index) - self.current_max_volumne.get_value(scenario_index)
            self.total_cost[scenario_index.global_id] = self.a * (storage ** self.b)

        return 0

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`."""
        return self.total_cost
    
    def aggregated_value(self):
        return self._scenario_aggregator.aggregate_1d(self.total_cost, ignore_nan=self.ignore_nan)
    

    @classmethod
    def load(cls, model, data):
        """Loads the recorder from a dictionary configuration."""

        parameter = load_parameter(model, data.pop("parameter"))
        current_max_volumne = load_parameter(model, data.pop("current_max_volumne"))
        
        # Remaining items in data are passed as kwargs to __init__
        return cls(model, parameter, current_max_volumne, **data)

ReservoirStorageExpansionCostRecorder.register()