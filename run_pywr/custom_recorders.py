import os
import sys
import numpy as np
import pandas as pd

from pywr.parameters import load_parameter
from pywr.recorders import NumpyArrayNodeRecorder, NodeRecorder, Aggregator, NumpyArrayStorageRecorder, NumpyArrayAbstractStorageRecorder, Recorder

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
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

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
        
        max_flow_param = self.node.max_flow

        areas = []
        for scenario_index in self.model.scenarios.combinations:
            areas.append(max_flow_param.area.get_value(scenario_index))
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        areas = pd.Series(np.array(areas), index=sc_index)

        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(max_flow_param.yield_per_area, axis=0)


        return crop_yield.mean(axis=0)


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
        
        max_flow_param = self.node.max_flow

        areas = []

        for scenario_index in self.model.scenarios.combinations:
            areas.append(max_flow_param.area.get_value(scenario_index))
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        #supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = demand.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        areas = pd.Series(np.array(areas), index=sc_index)

        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(max_flow_param.yield_per_area, axis=0)

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