import os
import sys
import numpy as np
import pandas as pd

from pywr.parameters import load_parameter, Parameter
from pywr.recorders import (NumpyArrayNodeRecorder, NodeRecorder, Aggregator, NumpyArrayStorageRecorder, NumpyArrayAbstractStorageRecorder, 
                            Recorder, hydropower_calculation, NumpyArrayParameterRecorder, BaseConstantParameterRecorder)
from pywr.recorders._recorders import NumpyArrayNodeRecorder

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

    # def __init__(self, model, node, observed, obs_freq=None, **kwargs):
    #     super(AbstractComparisonNodeRecorder, self).__init__(model, node, **kwargs)

    def __init__(self, model, node, observed, obs_freq=None, **kwargs):
        NumpyArrayNodeRecorder.__init__(self, model, node, **kwargs)

        self.observed = observed
        self._aligned_observed = None
        self.obs_freq = obs_freq

    def setup(self):
        # super(AbstractComparisonNodeRecorder, self).setup()
        NumpyArrayNodeRecorder.setup(self)
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
    """ Recorder evaluates the absolute percent bias between model and observed """
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

        # val = np.sum(obs-mod, axis=0)*100/np.sum(obs, axis=0)
        val = np.abs(np.sum(obs - mod, axis=0) * 100 / np.sum(obs, axis=0))

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


class RootMeanSquaredErrorStorageRecorder(AbstractComparisonStorageRecorder):
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


RootMeanSquaredErrorStorageRecorder.register()


class PercentBiasStorageRecorder(AbstractComparisonStorageRecorder):
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


PercentBiasStorageRecorder.register()


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

        return self._temporal_aggregator.aggregate_2d(crop_yield.values, axis=0, ignore_nan=self.ignore_nan) #crop_yield.mean(axis=0) 


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

        annual_hydropower = annual_hydropower.resample('Y').sum() # To get annual hydropower generation in MWh/year

        if self.factor is not None:
            annual_hydropower = annual_hydropower.multiply(self.factor, axis=0)

        return annual_hydropower

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

        return (rule - 4200) - outflow


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

        constraint = (rule - 4200) - outflow

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


# Delete everything from here! those are Mikiyas parameters and recorders.
from pywr.recorders import *
from pywr.nodes import Storage, Input, Node, Output
from pywr.parameters import (Parameter, load_parameter, InterpolatedVolumeParameter, MonthlyProfileParameter, ConstantParameter,
                             AggregatedParameter)
from pywr.parameters._hydropower import inverse_hydropower_calculation
import scipy.interpolate

class Rabi_water_allocation_origional(Parameter):
    def __init__(self, model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **kwargs):
        super().__init__(model, **kwargs)
        self.input_data = input_data
        self.Mangla_reservoir=Mangla_reservoir
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads
        
        self.Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}
        self.Sindh_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads}
        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads]
        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        
        self.Tarbela_reservoir=Tarbela_reservoir
        self.Indus_at_Chashma=Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla=Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela=Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range=percentage_range
        self.Filling_withdraw_fraction_Tarbela=Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla=Filling_withdraw_fraction_Mangla
        self.Eastern_rivers=Eastern_rivers
        self.JC_average_system_uses_1977_1982=JC_average_system_uses_1977_1982
        self.Average_System_use_Indus=Average_System_use_Indus
        self.KPK_Baloch_share=KPK_Baloch_share
        self.KPK_share_historical=KPK_share_historical
        self.Baloch_share_historical=Baloch_share_historical
        self.Below_Kotri=Below_Kotri
        self.Punjab_share_Indus_para_2_percent=Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus=System_losses_percent_Indus
        self.System_losses_JC=System_losses_JC

    def setup(self):
        super().setup()
        
    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = self.Tarbela_reservoir.max_volume/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
            
        if start_year < ts.year:
            if ts.month == 9 and ts.day > 29:
                self.Rabi_index = 0
                self.Rabi_season_start_date_index = ts.index
                water_balance_J_C_zone_output = water_balance_J_C_zone(self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC)
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"] 
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]

                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()

            Rabi_start_time = pd.to_datetime(str(ts.year) + '-09-29')
            Rabi_end_time = pd.to_datetime(str(ts.year + 1) + '-03-30') 
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
                #water used so far
                self.pubjab_abstructed_Rabi = {recorder:sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder:sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node:sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node:sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                
                if ts.day == 10 or ts.day == 20:
                    
                    self.Sindh_Channel_available_water = sum(self.Sindh_total_allocated_water[self.Rabi_index:])*43560
                    self.Punjab_J_C_Channel_available_water = sum(self.Punjab_J_C_total_allocated_water[self.Rabi_index:])
                    self.Punjab_Indus_Channel_available_water = sum(self.Punjab_Indus_total_allocated_water[self.Rabi_index:])
                    self.Rabi_index += 1
                    
                elif ts.day == days_in_month:
                    
                    self.Sindh_Channel_available_water = sum(self.Sindh_total_allocated_water[self.Rabi_index:])*43560
                    self.Punjab_J_C_Channel_available_water = sum(self.Punjab_J_C_total_allocated_water[self.Rabi_index:])
                    self.Punjab_Indus_Channel_available_water = sum(self.Punjab_Indus_total_allocated_water[self.Rabi_index:])
                    self.Rabi_index += 1
        
        #need the list of nodes in Punjab J-C and Indus Zone and Sindh Zone
        #check the next 183 days demand and if there less water reduce the demand by a certain fraction for those who are using more than the allocated amount
        #J-C outflow need to be tracked everytime and the value need to match with the allocated one
        
        #step1: collect all the recorders to get how much water is allocated 
        #step2: then create a number of 
             
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")
        
        return cls(model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data)  

Rabi_water_allocation_origional.register()


# class RootMeanSquaredErrorNodeRecorder_(NumpyArrayNodeRecorder):

#     def __init__(self, model, node, observed,index, **kwargs):
#         super(RootMeanSquaredErrorNodeRecorder_, self).__init__(model, node, **kwargs)
#         self.observed = pd.read_hdf(observed)[index]
#         self._aligned_observed = None
        

#     def setup(self):
#         super(RootMeanSquaredErrorNodeRecorder_, self).setup()
#         # Align the observed data to the model
#         self._aligned_observed = align_and_resample_dataframe(self.observed, self.model.timestepper.datetime_index)

#     def values(self):
#         mod = self.data
#         obs = self._aligned_observed
#         return np.sqrt(np.mean((obs-mod)**2, axis=0))

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
        
#         return cls(model, node, observed,index,**data)
    
# RootMeanSquaredErrorNodeRecorder_.register()


# class NashSutcliffeEfficiencyNodeRecorder_(NodeRecorder):

#     def __init__(self, model, node, observed, index, record_year, **kwargs):
#         super(NashSutcliffeEfficiencyNodeRecorder_, self).__init__(model, node, **kwargs)
#         self._aligned_observed = None
#         temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
#         factor = kwargs.pop('factor', 1.0)
#         self.factor = factor   
#         self._node = node
#         self.start=pd.to_datetime(str(record_year)+'-01-01 00:00:00')
#         self.end=pd.to_datetime(str(2009)+'-12-31 00:00:00')
#         self.observed = pd.read_hdf(observed)[index][ self.start:self.end]


#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         nts = len(self.model.timestepper)
#         self._data = np.zeros((nts))
#         self.nsc = np.zeros((nts))
#         # Align the observed data to the model
#         self._aligned_observed = self.observed

#     def reset(self):
#         self._data[:] = 0.0

#     def after(self):
#         ts = self.model.timestepper.current
#         #for i in range(self._data.shape[0]):
#         self._data[ts.index] = self._node.flow[0]
#         mod = self._data[-len(self._aligned_observed.values):]
#         obs = self._aligned_observed
#         obs_mean = np.mean(obs, axis=0)
#         self.nse_ = 1.0 - np.sum((obs-mod)**2, axis=0)/np.sum((obs-obs_mean)**2, axis=0)

#         self.nsc[ts.index] = self.nse_
    
        
#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         self.NSE_obj=np.zeros((1))
#         self.NSE_obj[0]=self.nse_
#         return self.NSE_obj

#     def to_dataframe(self):
#         """ Return a `pandas.DataFrame` of the recorder data
#         This DataFrame contains a MultiIndex for the columns with the recorder name
#         as the first level and scenario combination names as the second level. This
#         allows for easy combination with multiple recorder's DataFrames
#         """
#         index = self.model.timestepper.datetime_index
#         sc_index = self.model.scenarios.multiindex

#         return pd.DataFrame(data=np.array(self.nsc), index=index, columns=sc_index)

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
#         record_year = data.pop("record_year")
#         return cls(model, node, observed, index, record_year, **data)
    
# NashSutcliffeEfficiencyNodeRecorder_.register()


# class MeanSquareErrorNodeRecorder(NodeRecorder):

#     def __init__(self, model, node, observed, index, record_year, **kwargs):
#         super(MeanSquareErrorNodeRecorder, self).__init__(model, node, **kwargs)
#         self._aligned_observed = None
#         temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
#         factor = kwargs.pop('factor', 1.0)
#         self.factor = factor   
#         self._node = node
#         self.start=pd.to_datetime(str(record_year)+'-01-01 00:00:00')
#         self.end=pd.to_datetime(str(2009)+'-12-31 00:00:00')
#         self.observed = pd.read_hdf(observed)[index][ self.start:self.end]


#     def setup(self):
#         ncomb = len(self.model.scenarios.combinations)
#         nts = len(self.model.timestepper)
#         self._data = np.zeros((nts))
#         self.nsc = np.zeros((nts))
#         # Align the observed data to the model
#         self._aligned_observed = self.observed


#     def reset(self):
#         self._data[:] = 0.0

#     def after(self):
#         ts = self.model.timestepper.current
#         #for i in range(self._data.shape[0]):
#         self._data[ts.index] = self._node.flow[0]
#         mod = self._data[-len(self._aligned_observed.values):]
#         obs = self._aligned_observed
#         self.RMSE = np.mean((obs-mod)**2, axis=0)
        

#         return self.RMSE
        
#     def values(self):
#         """Compute a value for each scenario using `temporal_agg_func`."""
#         self.NSE_obj=np.zeros((1))
#         self.NSE_obj[0]=self.RMSE

#         return self.NSE_obj


#     def to_dataframe(self):
#         """ Return a `pandas.DataFrame` of the recorder data
#         This DataFrame contains a MultiIndex for the columns with the recorder name
#         as the first level and scenario combination names as the second level. This
#         allows for easy combination with multiple recorder's DataFrames
#         """
#         index = self.model.timestepper.datetime_index
#         sc_index = self.model.scenarios.multiindex

#         return pd.DataFrame(data=np.array(self.nsc), index=index, columns=sc_index)

#     @classmethod
#     def load(cls, model, data):
#         observed = data.pop("observed")   
#         index = data.pop("index")      
#         node = model._get_node_from_ref(model, data.pop("node"))    
#         record_year = data.pop("record_year")
#         return cls(model, node, observed, index, record_year, **data)
    
# MeanSquareErrorNodeRecorder.register()


class Simple_Irr_demand_calculator(Parameter):
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
                Irr_eff=1+(1-0.8)
                Irr_demand=Net_demand*(self.crop_area[x]*0.01)*self.Max_area*Irr_eff * 1e6 * 1e-3 * 1e-6
                
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
    
Simple_Irr_demand_calculator.register()


class Reservoir(Storage):
    def __init__(self, model, name, **kwargs):
        #where are you guys, the input and output nodes connected to self?
        #level = kwargs.pop('levels', None)
        volume = kwargs.pop('volumes', None)
        area = kwargs.pop('areas', None)
        rainfall= kwargs.pop('rainfall', None)
        evaporation= kwargs.pop('evaporation', None)
        weather_cost = kwargs.pop('weather_cost', -99999999)
        super().__init__(model, name, **kwargs)
        self._set_bathymetry(area, volume)
        self.rainfall_node = None
        self.evaporation_node = None
        
        if rainfall is not None:
            self._make_weather_nodes(model, rainfall, evaporation, weather_cost)
            
            
    def _set_bathymetry(self, areas,volumes):
        #self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)
        self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)
 

            
    def _make_weather_nodes(self, model, rainfall, evaporation,cost):
        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        rainfall_param = MonthlyProfileParameter(model, rainfall)
        evaporation_param = MonthlyProfileParameter(model, evaporation)

        # Assume rainfall/evap is mm/day
        # Need to convert:
        #   Mm2 -> m2
        #   mm/day -> m/day
        #   m3/day -> Mm3/day
        # TODO allow this to be configured
        const = ConstantParameter(model, 1e6 * 1e-3 * 1e-6)

        # Create the flow parameters multiplying area by rate of rainfall/evap
        rainfall_flow_param = AggregatedParameter(model, [rainfall_param, const, self.area],
                                                  agg_func='product')
        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, const, self.area],
                                                     agg_func='product')
        
        # Create the nodes to provide the flows
        rainfall_node = Input(model, '{}.rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param
        
        rainfall_node.cost = cost

        evporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evporation_node.max_flow = evaporation_flow_param

        evporation_node.cost = cost
        
        rainfall_node.connect(self)
        self.connect(evporation_node)
        self.rainfall_node = rainfall_node
        self.evaporation_node = evporation_node

        # Finally record these flows
        #self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node, name=f'__{rainfall_node.name}__:rainfall')
        #self.evaporation_recorder = NumpyArrayNodeRecorder(model, evporation_node, name=f'__{evporation_node.name}__:evaporation')    


class res_area_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, Area_factor, Reservoir_, **kwargs):
        super().__init__(model, **kwargs)
        self.Area_factor = Area_factor
        self.Reservoir_ = Reservoir_
        
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
    def value(self, timestep, scenario_index):    
        i = scenario_index.global_id
        
        Res_Volume = self.Reservoir_.volume[i]
        
        ts = self.model.timestepper.current        
        Res_Area=(Res_Volume**self.Area_factor)*1e-3
    
        return Res_Area
        

    @classmethod
    def load(cls, model, data):
        Area_factor = data.pop("Area_factor")
        Reservoir_ = data.pop("Reservoir")
        Reservoir_=model._get_node_from_ref(model, Reservoir_)
        return cls(model, Area_factor, Reservoir_, **data)
res_area_param.register()

class res_rainfall_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, url, index_col, column, Area_factor, Reservoir, **kwargs):
        super().__init__(model, **kwargs)
        self.url = pd.read_hdf(url)
        self.index_col = index_col
        self.column = column
        self.Area_factor = Area_factor
        self.Reservoir = Reservoir
        
        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
    def value(self, timestep, scenario_index):    
        i = scenario_index.global_id
        Res_Volume = self.Reservoir.volume[i]
        
        ts = self.model.timestepper.current
        
        month=ts.month
        year=ts.year
        
        rainfall=self.url[self.column][str(year)+"_"+str(month)]/ts.day
        
        Rain_volume=(Res_Volume**self.Area_factor)*rainfall*1e-3
        
    
        return Rain_volume
        

    @classmethod
    def load(cls, model, data):
        url = data.pop("url")
        index_col = data.pop("index_col")
        column = data.pop("column")
        Area_factor = data.pop("Area_factor")
        Reservoir = data.pop("Reservoir")
        Reservoir = model._get_node_from_ref(model, Reservoir)
        
        return cls(model, url, index_col, column, Area_factor, Reservoir, **data)
res_rainfall_param.register()

class res_vol_param(Parameter):
    """ A parameter triggers the maxmium hydropower capacity of a planned taking trigger year as an input
    ----------
    """
    def __init__(self, model, volume, **kwargs):
        super().__init__(model, **kwargs)
        self.volume = volume

        
    def setup(self):
        super().setup()
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        
        
    def value(self, timestep, scenario_index):        
        ts = self.model.timestepper.current
        year=1990
        years_=[int(x) for x in self.volume.keys()]
        volume_cap_year=sorted(i for i in years_ if i <= year)[0]
        volume=self.volume[str(volume_cap_year)]["volume"]

        return volume
        
    @classmethod
    def load(cls, model, data):
        volume = data.pop("volume")
        return cls(model, volume, **data)
res_vol_param.register()


class Rim_station_flow_forcast(Parameter):
    def __init__(self, model, historica_data, node_, probablity_table, rim_station, **kwargs):
        super().__init__(model, **kwargs)
        self.Flow_rim_station = pd.read_csv(historica_data)
        self.probablity_table = pd.read_csv(probablity_table)
        self.node_ = node_
        self.rim_station = rim_station
        self.percentage_range = 0.05

    def setup(self):
        super().setup()
        #self.node_ = self.model.recorders[self.node_]
        del self.probablity_table['Ten days']
        del self.probablity_table['Month']
        self.probablity_table = self.probablity_table.astype(float)
        self.Flow_rim_station.index = self.Flow_rim_station['Year']
        del self.Flow_rim_station['Year']
        self.prediction = np.zeros(18)
        self.end_of_Rabi = self.probablity_table.loc[17]
        self.end_of_Kharif =  self.probablity_table.loc[35]

    def calculate_matching_years(self, node_prev_flow, rim_station):
        #self.prediction should come from the node result for the past 6 month or the t-1 timestep (need to check)
        lower_bound = node_prev_flow - (node_prev_flow * self.percentage_range)
        upper_bound = node_prev_flow + (node_prev_flow * self.percentage_range)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@", lower_bound, upper_bound)
        #filter matching years
        filtered_df = self.Flow_rim_station[(self.Flow_rim_station[rim_station] >= lower_bound) & (self.Flow_rim_station[rim_station] <= upper_bound)]
        #average flows of matching years
        self.mean_value = filtered_df[rim_station].mean()
        

        return self.mean_value

    def probabilities_average_flow(self, end_of_season_flow, mean_value, season):
        # Calculate the absolute differences from the given value x

        if season == "Kharif":
            end_of_Rabi = self.probablity_table.loc[17]
            differences = abs(end_of_Rabi - mean_value)

            # Find the column with the minimum difference
            min_diff_column = differences.idxmin()
            
            prediction = self.probablity_table[min_diff_column][18:36]

        else:
            end_of_Kharif = self.probablity_table.loc[35]
            differences = abs(end_of_Kharif - mean_value)
            # Find the column with the minimum difference
            min_diff_column = differences.idxmin()
            prediction = self.probablity_table[min_diff_column][0:18]
        #should return the predected flow/storage_volume which should be a value rather than array
     
        return prediction

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if ts.year > 1991:
            if ts.month == 9 and ts.day > 28:
                season = "Rabi"
                node_prev_flow = np.sum(self.node_.data[ts.index - 180:ts.index])
                
                rim_station = self.rim_station + "_Rabi"
                mean_value = self.calculate_matching_years(node_prev_flow, rim_station)
                #mean_value = 14.847
                self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, season))

            elif ts.month == 3 and ts.day > 28: 
                
                season = "Kharif"
                node_prev_flow = np.sum(self.node_.data[ts.index - 180:ts.index])
                rim_station = self.rim_station + "_Kharif" 
                mean_value = self.calculate_matching_years(node_prev_flow, rim_station)
                #mean_value = 14.847
                self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, season))
            return self.prediction[0]
        else:
            return 0

    @classmethod
    def load(cls, model, data):
        historica_data = data.pop("historica_data")   
        node_ = load_recorder(model, data.pop("node"))      
        probablity_table = data.pop("probablity_table")
        rim_station = data.pop("rim_station")
        return cls(model, historica_data, node_, probablity_table, rim_station, **data)
Rim_station_flow_forcast.register()


class Reservoir_operation_5_years_average(Parameter):
    def __init__(self, model, historica_release, reservoir_name, **kwargs):
        super().__init__(model, **kwargs)
        self.historica_release = pd.read_hdf(historica_release)
        self.reservoir_name = reservoir_name

    def setup(self):
        super().setup()
        daily_mean_1994 = self.historica_release['1990-01-01':'1994-12-31'][self.reservoir_name]
        daily_mean_1998 = self.historica_release['1995-01-01':'1998-12-31'][self.reservoir_name]
        daily_mean_2003 = self.historica_release['1999-01-01':'2003-12-31'][self.reservoir_name]
        daily_mean_2008 = self.historica_release['2004-01-01':'2008-12-31'][self.reservoir_name]
        daily_mean_2013 = self.historica_release['2009-01-01':'2013-12-31'][self.reservoir_name]
        daily_mean_2018 = self.historica_release['2014-01-01':'2018-12-31'][self.reservoir_name]

        self.daily_mean_1994 = daily_mean_1994.groupby(daily_mean_1994.index.dayofyear).mean()
        self.daily_mean_1998 = daily_mean_1998.groupby(daily_mean_1998.index.dayofyear).mean()
        self.daily_mean_2003 = daily_mean_2003.groupby(daily_mean_2003.index.dayofyear).mean()
        self.daily_mean_2008 = daily_mean_2008.groupby(daily_mean_2008.index.dayofyear).mean()
        self.daily_mean_2013 = daily_mean_2013.groupby(daily_mean_2013.index.dayofyear).mean()
        self.daily_mean_2018 = daily_mean_2018.groupby(daily_mean_2018.index.dayofyear).mean()


    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        day = ts.dayofyear
        year = ts.year
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if year >= 1990 and year <= 1994:
            release = self.daily_mean_1994[day]
        elif year > 1994 and year <= 1998:
            release = self.daily_mean_1998[day]
        elif year > 1998 and year <= 2003:
            release = self.daily_mean_2003[day]
        elif year > 2003 and year <= 2008:
            release = self.daily_mean_2008[day]
        elif year > 2008 and year <= 2013:
            release = self.daily_mean_2013[day]
        else:
            release = self.daily_mean_2018[day]
        return release

    @classmethod
    def load(cls, model, data):
        historica_release = data.pop("historica_release")   
        reservoir_name = data.pop("reservoir_name")    
        return cls(model, historica_release, reservoir_name, **data)
Reservoir_operation_5_years_average.register()

class Reservoir_operation_10_years_average(Parameter):
    def __init__(self, model, historica_release, reservoir_name, **kwargs):
        super().__init__(model, **kwargs)
        self.historica_release = pd.read_hdf(historica_release)
        self.reservoir_name = reservoir_name

    def setup(self):
        super().setup()
        daily_mean_1994 = self.historica_release['1990-01-01':'1998-12-31'][self.reservoir_name]
        daily_mean_2003 = self.historica_release['1999-01-01':'2008-12-31'][self.reservoir_name]
        daily_mean_2013 = self.historica_release['2009-01-01':'2018-12-31'][self.reservoir_name]

        self.daily_mean_1994 = daily_mean_1994.groupby(daily_mean_1994.index.dayofyear).mean()
        self.daily_mean_2003 = daily_mean_2003.groupby(daily_mean_2003.index.dayofyear).mean()
        self.daily_mean_2013 = daily_mean_2013.groupby(daily_mean_2013.index.dayofyear).mean()

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current
        day = ts.dayofyear
        year = ts.year
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)

        if year >= 1990 and year <= 1998:
            release = self.daily_mean_1994[day]
        elif year > 1998 and year <= 2008:
            release = self.daily_mean_2003[day]
        else:
            release = self.daily_mean_2013[day]
        return release 

    @classmethod
    def load(cls, model, data):
        historica_release = data.pop("historica_release")   
        reservoir_name = data.pop("reservoir_name")    
        return cls(model, historica_release, reservoir_name, **data)
Reservoir_operation_10_years_average.register()


class Reservior_operation_matching_year(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column


    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column] 

        

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current


        i = scenario_index.global_id 
        current_water_level = self.Storage_node.value(timestep, scenario_index)
        current_inflow = self.inflow.value(timestep, scenario_index)


        # Calculate distance
        self.historical_flow_df["distance"]  = np.sqrt((self.Res_level - current_water_level)**2 + (self.Res_inflow - current_inflow)**2)

        # Find the row with the minimum distance
        nearest_row = self.historical_flow_df .loc[self.historical_flow_df["distance"].idxmin()]

        # Extract release from the nearest row
        nearest_release = nearest_row[self.outflow_column]
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("Historical_release_wl")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    

        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = load_parameter(model, data.pop("inflow_node")) 

        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **data)
Reservior_operation_matching_year.register()


class Reservior_operation_matching_year_2(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column


    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column] 

        

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current


        i = scenario_index.global_id 
        current_water_level = self.Storage_node.value(timestep, scenario_index)

        current_inflow = self.Res_inflow[str(ts.datetime.date())] 
        
        


        # Calculate distance
        self.historical_flow_df["distance"]  = np.sqrt((self.Res_level - current_water_level)**2 + (self.Res_inflow - current_inflow)**2)

        # Find the row with the minimum distance
        nearest_row = self.historical_flow_df.loc[self.historical_flow_df["distance"].idxmin()]

        

        # Extract release from the nearest row
        nearest_release = nearest_row[self.outflow_column]
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("url")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    
        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = model._get_node_from_ref(model, data.pop("inflow_node")) 
        

        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, **data)
Reservior_operation_matching_year_2.register()


###################################################################################################################
###################################################################################################################

class Reservior_operation_matching_year_average(Parameter):
    def __init__(self, model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, number_years, **kwargs):
        super().__init__(model, **kwargs)
        self.historical_flow_df = Historical_release_wl
        self.Storage_node = Storage_node
        self.inflow = inflow
        self.inflow_column = inflow_column
        self.outflow_column = outflow_column
        self.reservoir_level_column = reservoir_level_column
        self.number_years = number_years

    def setup(self):
        super().setup()
        self.historical_flow_df = pd.read_hdf(self.historical_flow_df)
        self.Res_inflow = self.historical_flow_df[self.inflow_column]
        self.Res_outflow = self.historical_flow_df[self.outflow_column]
        self.Res_level = self.historical_flow_df[self.reservoir_level_column]
        
        mask = ~((self.historical_flow_df.index >= '1997-10-01') & (self.historical_flow_df.index <= '2003-04-01'))
        
        self.Res_inflow_MG_NF = self.historical_flow_df[self.inflow_column][mask]
        self.Res_outflow_MG_NF = self.historical_flow_df[self.outflow_column][mask]
        self.Res_level_MG_NF = self.historical_flow_df[self.reservoir_level_column][mask]
        
        self.Res_inflow_MG_LF = self.historical_flow_df[self.inflow_column]['01/10/1997':'01/04/2003']
        self.Res_outflow_MG_LF = self.historical_flow_df[self.outflow_column]['01/10/1997':'01/04/2003']
        self.Res_level_MG_LF = self.historical_flow_df[self.reservoir_level_column]['01/10/1997':'01/04/2003']

    def value(self, timestep, scenario_index):
        ts = self.model.timestepper.current

        i = scenario_index.global_id 
        
        if (ts.year > 1997) and (ts.year < 2003):
            Res_level_n = self.Res_level_MG_LF[self.Res_level_MG_LF.index.month == ts.month]
            Res_inflow_n = self.Res_inflow_MG_LF[self.Res_inflow_MG_LF.index.month == ts.month]
            
            current_water_level = self.Storage_node.value(timestep, scenario_index)
            #current_inflow = self.inflow.flow
            current_inflow = self.Res_inflow[str(ts.datetime.date())] 
            
            # Calculate distance
            self.historical_flow_df["distance"]  = np.sqrt((Res_level_n - current_water_level)**2 + (Res_inflow_n - current_inflow)**2)
    
            historical_flow_df_sorted = self.historical_flow_df.sort_values(by=['distance'])
            nearest_release = historical_flow_df_sorted.iloc[:4][self.outflow_column].mean()
        
        else:   
        
            Res_level_n = self.Res_level_MG_NF[self.Res_level_MG_NF.index.month == ts.month]
            Res_inflow_n = self.Res_inflow_MG_NF[self.Res_inflow_MG_NF.index.month == ts.month]
            
            current_water_level = self.Storage_node.value(timestep, scenario_index)
            #current_inflow = self.inflow.flow
            current_inflow = self.Res_inflow[str(ts.datetime.date())] 
            
            # Calculate distance
            self.historical_flow_df["distance"]  = np.sqrt((Res_level_n - current_water_level)**2 + (Res_inflow_n - current_inflow)**2)
    
            historical_flow_df_sorted = self.historical_flow_df.sort_values(by=['distance'])
            nearest_release = historical_flow_df_sorted.iloc[:self.number_years][self.outflow_column].mean()
        
        return nearest_release


    @classmethod
    def load(cls, model, data):

        Historical_release_wl = data.pop("url")
        reservoir_level_column = data.pop("reservoir_level_column")    
        inflow_column = data.pop("inflow_column") 
        outflow_column = data.pop("outflow_column")    
        Storage_node = load_parameter(model, data.pop("Storage_node"))
        inflow = model._get_node_from_ref(model, data.pop("inflow_node")) 
        number_years =  data.pop("number_years")  
        
        return cls(model, Historical_release_wl, reservoir_level_column, inflow_column, outflow_column, Storage_node, inflow, number_years, **data)

Reservior_operation_matching_year_average.register()


class River_seasonal_loss(Parameter):
    def __init__(self, model, loss_early_Rabi, loss_late_Rabi, loss_early_Kharif, loss_late_Khraif, loss_node, **kwargs):
        super().__init__(model, **kwargs)
        self.loss_node=loss_node 
        self.loss_early_Rabi = loss_early_Rabi
        self.loss_late_Rabi = loss_late_Rabi
        self.loss_early_Kharif = loss_early_Kharif
        self.loss_late_Khraif = loss_late_Khraif


    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        node_flow=self.loss_node.prev_flow[i]
        ts = self.model.timestepper.current
        
        
        
        # the loss values are in fraction not in percentage
        if ts.month >= 10 and ts.month <= 12:
            
            if self.loss_early_Rabi < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_early_Rabi)
        elif ts.month >= 1 and ts.month <= 3:
            if self.loss_late_Rabi < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_late_Rabi)
        elif ts.month >= 4 and ts.month <= 6:
            if self.loss_early_Kharif < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_early_Kharif)
        elif ts.month >= 7 and ts.month <= 9:
            if self.loss_late_Khraif < 0:
                loss = 0
            else:
                loss = node_flow*(self.loss_late_Khraif)
            
        
        return loss
            
    @classmethod
    def load(cls, model, data):
        loss_early_Rabi = data.pop("loss_early_Rabi")
        loss_late_Rabi = data.pop("loss_late_Rabi")
        loss_early_Kharif = data.pop("loss_early_Kharif")
        loss_late_Khraif = data.pop("loss_late_Khraif")
        loss_node=model._get_node_from_ref(model, data.pop("Loss_node"))
        
        return cls(model, loss_early_Rabi, loss_late_Rabi, loss_early_Kharif, loss_late_Khraif, loss_node, **data)
        
River_seasonal_loss.register()


class Rim_station_flow_forcast(Parameter):
    def __init__(self, model, Punjab_channel_heads_node, reservoir_test, Historical_flow_Rim_station, IRSA_probablity_table, rim_station, node_, punjab_demand_nodes, sindh_demand_nodes, **kwargs):
        super().__init__(model, **kwargs)
        self.Historical_flow_Rim_station = Historical_flow_Rim_station
        self.IRSA_probablity_table = IRSA_probablity_table
        self.node_ = node_
        self.reservoir_test = reservoir_test
        self.Punjab_channel_heads_node = Punjab_channel_heads_node
        self.rim_station = rim_station
        self.punjab_demand_nodes = punjab_demand_nodes 
        self.sindh_demand_nodes = sindh_demand_nodes
        self.percentage_range = 0.05

    def setup(self):
        super().setup()
        self.Rabi_index = 0
        self.Kharif_index = 0
        self.Rim_station_data = pd.HDFStore(self.Historical_flow_Rim_station)
        self.Flow_rim_station = self.Rim_station_data['/Historical_flow']
        self.probablity_table = pd.HDFStore(self.IRSA_probablity_table)
        self.prediction = np.zeros(18)

    def calculate_matching_years(self, node_prev_flow, rim_station):
        lower_bound = node_prev_flow - (node_prev_flow * self.percentage_range)
        upper_bound = node_prev_flow + (node_prev_flow * self.percentage_range)
        filtered_df = self.Flow_rim_station[(self.Flow_rim_station[rim_station] >= lower_bound) & (self.Flow_rim_station[rim_station] <= upper_bound)]
        mean_value = filtered_df[rim_station].mean()
        return mean_value

    def probabilities_average_flow(self, end_of_season_flow, mean_value, season, rim_stations):
        if season == "Kharif":
            probablity_table = self.probablity_table[rim_stations]
            end_of_Rabi = probablity_table.loc[0:17].sum()
            differences = abs(end_of_Rabi - mean_value)
            min_diff_column = differences.idxmin()
            prediction = np.array(probablity_table[min_diff_column][18:36])
            
        else:
            
            probablity_table = self.probablity_table[rim_stations]
            end_of_Kharif = probablity_table.loc[18:35].sum()
            differences = abs(end_of_Kharif - mean_value)
            min_diff_column = differences.idxmin()
            prediction = np.array(probablity_table[min_diff_column][0:18])
        return prediction


    def value(self, timestep, scenario_index):
        
        ts = self.model.timestepper.current
        start_year = self.model.timestepper.start.year
        days_in_month = timestep.period.days_in_month
        #there are two seasons Rabi (from April 1 to Sep 20) and Kharif (Oct to March)
        self.basin_wide_Rabi_prediction = []
        Rabi_start_time = pd.to_datetime(str(ts.year - 1) + '-09-29')
        Rabi_end_time = pd.to_datetime(str(ts.year) + '-03-30') 
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index

        #get_all_values
        #get_value
        if start_year < ts.year:
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
                pass

            if ts.month == 9 and ts.day > 29:
                self.Kharif_index = 0
                self.season = "Kharif"
                self.Kharif_prediction = {}
                for rim_station in self.rim_station:
                    self.node = self.node_[rim_station]
                    node_prev_flow = np.sum(self.node.data[ts.index - 180:ts.index])
                    rim_station_seasonal = rim_station + "_Kharif"
                    mean_value = self.calculate_matching_years(node_prev_flow, rim_station_seasonal)
                    self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, self.season, rim_station))
                    self.Kharif_prediction[rim_station] = self.prediction
                
                var_len = len(self.Kharif_prediction[list(self.Kharif_prediction.keys())[0]])
                self.basin_wide_Kharif_prediction = [sum([self.Kharif_prediction[rim_stat][index] for rim_stat in self.Kharif_prediction.keys()]) for index in range(var_len)]
                
            elif ts.month == 3 and ts.day > 30:
                self.Rabi_index = 0
                self.season = "Rabi"
                self.Rabi_prediction = {}
                for rim_station in self.rim_station:
                    self.node = self.node_[rim_station]
                    node_prev_flow = np.sum(self.node.data[ts.index - 180:ts.index])
                    rim_station_seasonal = rim_station + "_Rabi" 
                    mean_value = self.calculate_matching_years(node_prev_flow, rim_station_seasonal)
                    self.prediction = list(self.probabilities_average_flow(node_prev_flow, mean_value, self.season, rim_station))
                    self.Rabi_prediction[rim_station] = self.prediction
                var_len = len(self.Rabi_prediction[list(self.Rabi_prediction.keys())[0]])
                self.basin_wide_Rabi_prediction = [sum([self.Rabi_prediction[rim_stat][index] for rim_stat in self.Rabi_prediction.keys()]) for index in range(var_len)]
        
            
            #need to convert rim stations prediction to basin wide prediction
            #Update the predicted flow
            deficit_percentage_sindh = 0.552174606
            deficit_percentage_punjab = 0.447825394
            
            punjab_demand = sum([demand_node.get_max_flow(scenario_index) for demand_node in self.punjab_demand_nodes])
            sindh_demand = sum([demand_node.get_max_flow(scenario_index)  for demand_node in self.sindh_demand_nodes])


            if ts.month >= 4 and ts.month <= 9:
                #self.Rabi_index = "Rabi"
                total_Rabi_loss = 0
                Rabi_allocation_under_normal_condition = 2734261.2
                Rabi_punjab_share_under_normal_condition = 2734261.6*0.4 
                Rabi_sindh_share_under_normal_condition = 2734261.6*0.3 
                if ts.day == 10 or ts.day == 20:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - 10: ts.index]) for inflow in self.node_])
                    self.basin_wide_Rabi_prediction[self.Rabi_index] = observed_value 
                    self.Rabi_index += 1
                elif ts.day == days_in_month:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - days_in_month - 20: ts.index]) for inflow in self.node_])
                    self.basin_wide_Rabi_prediction[self.Rabi_index] = observed_value 
                    self.Rabi_index += 1
                Rabi_deficit_volume = (Rabi_allocation_under_normal_condition - (sum(self.basin_wide_Rabi_prediction) - total_Rabi_loss))

                if Rabi_deficit_volume > 0:
    
                    Rabi_punjab_deficit_fraction = (Rabi_deficit_volume*deficit_percentage_punjab/Rabi_punjab_share_under_normal_condition)
                    Rabi_sindh_deficit_fraction = (Rabi_deficit_volume*deficit_percentage_sindh/Rabi_sindh_share_under_normal_condition)
                    
                    Rabi_sindh_deficit_volume = sindh_demand*Rabi_sindh_deficit_fraction
                    Rabi_punjab_deficit_volume = punjab_demand*Rabi_punjab_deficit_fraction

                    for demand_node in self.punjab_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/punjab_demand)*Rabi_punjab_deficit_volume
                    for demand_node in self.sindh_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/sindh_demand)*Rabi_sindh_deficit_volume 


            elif ts.month < 4 and ts.month > 9:
                total_Kharif_loss = 0
                Kharif_allocation_under_normal_condition = 1514145.6
                Kharif_punjab_share_under_normal_condition = 1514145.6*0.4 
                Kharif_sindh_share_under_normal_condition = 1514145.6*0.3 
                if ts.day == 10 or ts.day == 20:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - 10: ts.index]) for inflow in self.node_])
                    self.basin_wide_Kharif_prediction[self.Kharif_index] = observed_value
                    self.Kharif_index += 1
                    
                elif ts.day == days_in_month:
                    observed_value = sum([sum(self.node_[inflow].data[ts.index - days_in_month - 20: ts.index]) for inflow in self.node_])
                    self.basin_wide_Kharif_prediction[self.Kharif_index] = observed_value
                    self.Kharif_index += 1
                Kharif_deficit_volume = (Kharif_allocation_under_normal_condition - (sum(self.basin_wide_Kharif_prediction) - total_Kharif_loss))

                if Kharif_deficit_volume > 0:

                    Kharif_punjab_deficit_fraction = (Kharif_deficit_volume*deficit_percentage_punjab/Kharif_punjab_share_under_normal_condition)
                    Kharif_sindh_deficit_fraction = (Kharif_deficit_volume*deficit_percentage_sindh/Kharif_sindh_share_under_normal_condition) 
                    
                    Kharif_sindh_deficit_volume = sindh_demand*Kharif_sindh_deficit_fraction
                    Kharif_punjab_deficit_volume = punjab_demand*Kharif_punjab_deficit_fraction

                    for demand_node in self.punjab_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/punjab_demand)*Kharif_punjab_deficit_volume
                    for demand_node in self.sindh_demand_nodes:
                        demand_node.max_flow = (demand_node.get_max_flow(scenario_index)/sindh_demand)*Kharif_sindh_deficit_volume 
        
        return 0


    @classmethod
    def load(cls, model, data):
        Historical_flow_Rim_station = data.pop("Historical_flow_Rim_station")
        IRSA_probablity_table = data.pop("IRSA_probablity_table")
        rim_station_observed = data.pop("rim_station_observed")
        reservoir_test = model._get_node_from_ref(model, data.pop("Reservoir_name"))
        
        node_ = {node : load_recorder(model, rim_station_observed[node]) for node in rim_station_observed}
        rim_station = [node for node in rim_station_observed]
        punjab_demand_nodes = [model._get_node_from_ref(model, node_name) for node_name in data["punjab_demand"]] 
        Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in data.pop("punjab_demand")}
        sindh_demand_nodes = [model._get_node_from_ref(model, node_name) for node_name in data.pop("sindh_demand")]
        return cls(model, Punjab_channel_heads_node, reservoir_test, Historical_flow_Rim_station, IRSA_probablity_table, rim_station, node_, punjab_demand_nodes, sindh_demand_nodes, **data)
Rim_station_flow_forcast.register()

class Distribution_plan_parameter(Parameter):
    def __init__(self, model, distribution_plan, **kwargs):
        super().__init__(model, **kwargs)
        self.distribution_plan = distribution_plan 
        self.month_index = {1: "January",2: "February",3: "March",4: "April",5: "May",6: "June",7: "July",8: "August",9: "September",10: "October",11: "November",12: "December"}

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        month_name = self.month_index[ts.month] 

        if ts.day <= 10:
            key = month_name+"-1"
            return self.distribution_plan[key]
            
        elif ts.day > 10 and ts.day <= 20:
            key = month_name+"-2"
            return self.distribution_plan[key]
        else:
            key = month_name+"-3"
            return self.distribution_plan[key]
            
    @classmethod
    def load(cls, model, data):
        distribution_plan = data.pop("distribution_plan")
        return cls(model, distribution_plan, **data)      
Distribution_plan_parameter.register()

def water_balance_J_C_zone(input_data,Initial_Storage_Mangla,Maximum_Storage_Mangla,Initial_Storage_Tarbela,Maximum_Storage_Tarbela,Indus_at_Chashma,Storage_Dep_at_end_of_season_Mangla,Storage_Dep_at_end_of_Season_Tarbela,start_date,percentage_range,Filling_withdraw_fraction_Tarbela,Filling_withdraw_fraction_Mangla,Eastern_rivers,JC_average_system_uses_1977_1982,Average_System_use_Indus,KPK_Baloch_share,KPK_share_historical,Baloch_share_historical,Below_Kotri,Punjab_share_Indus_para_2_percent,System_losses_percent_Indus,System_losses_JC):
    #load input dataframe
    season = "Rabi"
    input_data = pd.HDFStore(input_data)
    observed_flow=input_data['/observed_flow']
    rim_station = ['Kabul_Noshehra', 'Indus_Tarbela', 'Jhelum_Mangla', 'Chenab_Marala']
    J_C_rim_stations_name = rim_station[2:4]
    Indus_rim_stations_name = rim_station[:2]
    observed_flow_index = observed_flow.index.get_loc(start_date)

    ##############################################Input_from_model############################################

    ##############################################Input_from_model############################################
    
    day_month_correction = [1, 1, 1.1, 1, 1, 1, 1, 1, 1.1, 1, 1, 1.1, 1, 1, 0.8, 1, 1, 1.1]
    CUMCS_MAC_conversion_factor = 0.01983471
    J_C_1977_88 = [49.6,45.7,42.5,39.1,37.1,35.9,36,34.4,25.7,12.7,14.6,20.9,27.7,33.2,29.5,32.9,35.7,35.6]
    J_C_1977_88_MAF = [CUMCS_MAC_conversion_factor*day_month_correction[index]*J_C_1977_88[index] for index in range(len(J_C_1977_88))]
 
    def convert_to_MAF(data):
        for station, data_dict in data.items():
            for key, arr in data_dict.items():
                data[station][key] = arr / 43560
        return data
    
    Rim_prediction = convert_to_MAF({rm : IRSA_prediction__Rabi(rm, input_data, season, observed_flow_index, percentage_range) for rm in rim_station})
    J_C_Rim_stations_prediction = {key: Rim_prediction[key] for key in J_C_rim_stations_name}
    Indus_Rim_stations_prediction = {key: Rim_prediction[key] for key in Indus_rim_stations_name}
    Min_JC = {}
    Max_JC = {}
    
    
    if season == "Rabi":
    
        J_C_Shortage = {}
        Punjab_J_C_Canal_Wdls_Outflow_RQBS = {}
        Sindh_Channel_dis_df = {}
        Punjab_J_C_Channel_dis_df = {}
        Punjab_Indus_Channel_dis_df = {}
        scenarios = ["Maximum", "Minimum"]
        for scenario in scenarios:
            
            ############################JJJJCCC################################################
            #Forecast of Inflows
            System_Outflow_RQBS_min_max = {}
            Jhelum_at_Mangla = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario])
            Chenab_at_Marala = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario])
            Rabi_Inflows_J_C_Command = Jhelum_at_Mangla + Chenab_at_Marala + sum(Eastern_rivers)
            
            
 
            Storage_Available = Initial_Storage_Mangla
            #Storage_Dep_at_end_of_Season_Mangla = 100
            Storage_Release = Storage_Available - (100 - Storage_Dep_at_end_of_season_Mangla)*Maximum_Storage_Mangla
            System_Inflows = Storage_Release + Rabi_Inflows_J_C_Command
            Total_Availability_JC = System_Inflows * (1 - System_losses_JC/100)
            
            ##############################IIINNNN#################################################

            Indus_at_Tarbela = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario])
            Kabul_at_Nowshehra = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario])
            Total_Indus = Indus_at_Tarbela + Kabul_at_Nowshehra + Indus_at_Chashma
            
            Storage_Available_Tarbela = Initial_Storage_Tarbela
            Storage_Release_Tarbela = Initial_Storage_Tarbela - (100 - Storage_Dep_at_end_of_Season_Tarbela)*Maximum_Storage_Tarbela
            KPK_share = [day_month_correction[index]*KPK_share_historical[index]*0.01983471 for index in range(len(KPK_share_historical))]
            Baloch_share = [day_month_correction[index]*Baloch_share_historical[index]*0.01983471 for index in range(len(KPK_share_historical))]
            System_losses_Indus = 1 - System_losses_percent_Indus/100
            
            
            Indus_know_parameters = System_losses_Indus*(Storage_Release_Tarbela + Total_Indus) - KPK_Baloch_share - Below_Kotri
            Numerator = Total_Availability_JC*Average_System_use_Indus - JC_average_system_uses_1977_1982*Indus_know_parameters
            Denominator = System_losses_Indus*JC_average_system_uses_1977_1982 + Average_System_use_Indus
            J_C_Outflow = Numerator/Denominator
            
            """
            System_Inflows_Indus = Total_Indus + sum(J_C_Outflow) + Storage_Release_Tarbela
            System_losses_Indus = System_losses_percent_Indus*System_Inflows_Indus/100
            """
            
            ############################JJJJCCC################################################
            JC_Canal_Availability = Total_Availability_JC - J_C_Outflow  #Where is this value comming from?
            J_C_Shortage = (1 - JC_Canal_Availability/JC_average_system_uses_1977_1982)*100
    
            ############################JJJJCCC################################################
            Live_content_MAF = [Initial_Storage_Mangla]  #in MAC
            Initial_Storage_temp = Initial_Storage_Mangla
            
            for vol in Filling_withdraw_fraction_Mangla:
                if Initial_Storage_temp - Storage_Release*vol/100 < 0:
                    Live_content_MAF.append(0)
                else:
                    release = (Storage_Release*vol/100)
                    Initial_Storage_temp = round(Initial_Storage_temp - release, 3)
                    Live_content_MAF.append(Initial_Storage_temp)
            
            
            Mangla_outflow = []
            for index in range(18):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][index] - (Live_content_MAF[index+1]-Live_content_MAF[index])
                
                Mangla_outflow.append(release)

            System_Inflow = [Mangla_outflow[index] + Eastern_rivers[index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][index] for index in range(len(Mangla_outflow))]
            Net_inflow_after_loss = [-1*inflow*(System_losses_JC/100) for inflow in System_Inflow]
            Net_inflow = [System_Inflow[index] - Net_inflow_after_loss[index] for index in range(len(Net_inflow_after_loss))]
            
            Punjab_J_C_Canal_Wdls = []
            Punjab_J_C_Canal_Wdls = [
                Net_inflow[index] if (1 - J_C_Shortage * 0.01) * J_C_1977_88_MAF[index] > Net_inflow[index] 
                else (1 - J_C_Shortage * 0.01) * J_C_1977_88_MAF[index]
                for index in range(len(Net_inflow))
            ]
            System_Outflow_RQBS = [Net_inflow[index] - Punjab_J_C_Canal_Wdls[index] for index in range(len(Punjab_J_C_Canal_Wdls))]

            
            
            ############################JJJJCCC################################################

            ##############################IIINNNN#################################################
            #JC_outflow is part of the below equation that need to be fixed
            
            #JC_outflow is part of the below equation that need to be fixed
            System_Inflows_Indus = Total_Indus + J_C_Outflow + Storage_Release_Tarbela
            System_losses_Indus_volume = System_Inflows_Indus*System_losses_percent_Indus/100
            
            Total_Availability_Indus = System_Inflows_Indus - System_losses_Indus_volume
            Canal_Availability_Indus = Total_Availability_Indus - Below_Kotri
            Punjab_Sindh_share_Indus = Canal_Availability_Indus - KPK_Baloch_share
            Shortage = (1 - Punjab_Sindh_share_Indus/Average_System_use_Indus)*100
            
            ##############################IIINNNN#################################################
        
            
            
            ##############################IIINNNN#######################################################################
            Live_content_MAF = [Initial_Storage_Tarbela]  #in MAC
            Initial_Storage_temp = Initial_Storage_Tarbela

            for vol in Filling_withdraw_fraction_Tarbela:
                if Initial_Storage_temp - Storage_Release_Tarbela*vol/100 < 0:
                    Live_content_MAF.append(0)
                else:
                    release = (Storage_Release_Tarbela*vol/100)
                    Initial_Storage_temp = round(Initial_Storage_temp - release, 3)
                    Live_content_MAF.append(Initial_Storage_temp)
                                        
            Tarbela_outflow = []
            for index in range(len(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario])):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][index]-(Live_content_MAF[index+1]-Live_content_MAF[index])
                Tarbela_outflow.append(release)
            System_Inflow = [Tarbela_outflow[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][index] + System_Outflow_RQBS[index] for index in range(len(Tarbela_outflow))]
            Net_inflow_after_loss_Indus = [-1*inflow*System_losses_percent_Indus/100 for inflow in System_Inflow]
            Net_inflow_Indus = [System_Inflow[x] + Net_inflow_after_loss_Indus[x] for x in range(len(Net_inflow_after_loss_Indus))]
            
            Proposed_Canal_Wdls_Indus = Net_inflow_Indus    
            System_Outflow_DS_Kotri = [Net_inflow_Indus[index]-Proposed_Canal_Wdls_Indus[index] for index in range(len(Net_inflow_Indus))]

            
            
            Total_Share_Punjab_Sindh = [Proposed_Canal_Wdls_Indus[index] - KPK_share[index] - Baloch_share[index] for index in range(len(KPK_share))]
            #condition to implement the Para2 para14b and 
            Sindh_share_Indus_para_2_percent = [100 - Punjab_share_Indus_para_2_percent[index] for index in range(len(Punjab_share_Indus_para_2_percent))]
            
            
            Punjab_Indus_Canal_Wdls = [Total_Share_Punjab_Sindh[index]*Punjab_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]
            Sindh_share_Canal_Wdls  = [Total_Share_Punjab_Sindh[index]*Sindh_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]

            ##############################IIINNNN#######################################################################
            
            Sindh_Channel_dis_df[scenario] = input_data['Sindh_Channel_dis_plan_percentage'][0:18].mul(Sindh_share_Canal_Wdls, axis=0)
            Punjab_J_C_Channel_dis_df[scenario] = input_data['J_C_Channel_dis_plan_percentage'][0:18].mul(Punjab_J_C_Canal_Wdls, axis=0)
            Punjab_Indus_Channel_dis_df[scenario] = input_data['Indus_Channel_dis_plan_percentage'][0:18].mul(Punjab_Indus_Canal_Wdls, axis=0)
            System_Outflow_RQBS_min_max[scenario] = System_Outflow_RQBS 

            
        output = {}
        Sindh_Channel_dis_df_likely = (Sindh_Channel_dis_df[scenarios[0]] + Sindh_Channel_dis_df[scenarios[1]]) / 2
        Punjab_J_C_Channel_dis_df_likely = (Punjab_J_C_Channel_dis_df[scenarios[0]] + Punjab_J_C_Channel_dis_df[scenarios[1]]) / 2
        Punjab_Indus_Channel_dis_df_likely = (Punjab_Indus_Channel_dis_df[scenarios[0]] + Punjab_Indus_Channel_dis_df[scenarios[1]]) / 2
        RQBS_Canal_Outflow_likely =  [sum(x) / len(x) for x in zip(*System_Outflow_RQBS_min_max.values())]
        
 
        output["Sindh_Channel_dis_df_likely"] = Sindh_Channel_dis_df_likely
        output["Punjab_J_C_Channel_dis_df_likely"] = Punjab_J_C_Channel_dis_df_likely
        output["Punjab_Indus_Channel_dis_df_likely"] = Punjab_Indus_Channel_dis_df_likely
        output["RQBS_Canal_Outflow_likely"] = RQBS_Canal_Outflow_likely
        
            

        #RQBS_Canal_Outflow_likely.to_csv("RQBS_Canal_Outflow_likely")
    elif season == "Khrif":
        #ToDo Khrif implementation
        pass
    
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ImplementationPlan@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    """
    1. update the IRSA prediction with observed flow and reservoir active Storage
    2. redo the calculation to get how much water is avaiable
    3. check if there is enough water to satisfy the provinital water allocaiton and make sure that the provinces does not use above the allocated water 
    4. curtail the demand if the available water is below the remaining demand (know which to curtail)
    """
    
    """
    forcing the model to release the RQBS flow to Indus basin.
    """
    input_data.close()
    return output

def IRSA_prediction__Rabi(rim_station_name, input_df, season, index, percentage_range):
    
    observed_flow_values = input_df['/observed_flow'][rim_station_name]
    probablity_table = input_df[rim_station_name]
    Flow_rim_station = input_df['/Historical_flow']
    probablity_column = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    def calculate_matching_years(node_prev_flow, rim_station):
        lower_bound = node_prev_flow * (1 - percentage_range[rim_station_name])
        upper_bound = node_prev_flow * (1 + percentage_range[rim_station_name])
        filtered_df = Flow_rim_station[(Flow_rim_station[rim_station] >= lower_bound) & (Flow_rim_station[rim_station] <= upper_bound)]
        
        mean_value = filtered_df[rim_station].mean()
        return mean_value

    def probabilities_average_flow(end_of_season_flow, mean_value, season):
        min_max = {'Minimum': -0.1, 'Maximum': 0.1}
        prediction_min_max = {}
        if season == "Kharif":
            end_of_Rabi = probablity_table.loc[18:36].sum()
            differences = abs(end_of_Rabi - mean_value)
            
            for key in min_max:
                val = min_max[key]
                if differences.idxmin() == "SYN Max":
                    probablity_val = 0.05
                elif differences.idxmin() == "SYN Min":
                    probablity_val = 0.95
                else:
                    probablity_val = differences.idxmin()
                
                min_diff_column = probablity_val + val
                closest_number = min(probablity_column, key=lambda x: abs(x - min_diff_column))
                prediction = np.array(probablity_table[closest_number][0:18])
                prediction_min_max[key] = prediction

        else:
            end_of_Kharif = probablity_table.loc[0:18].sum()
            differences = abs(end_of_Kharif - mean_value)
            
            for key in min_max:
                val = min_max[key]
                if differences.idxmin() == "SYN Max":
                    probablity_val = 0.05
                elif differences.idxmin() == "SYN Min":
                    probablity_val = 0.95
                else:
                    probablity_val = differences.idxmin()
                
                min_diff_column = probablity_val + val
                closest_number = min(probablity_column, key=lambda x: abs(x - min_diff_column))
                prediction = np.array(probablity_table[closest_number][18:36])
                prediction_min_max[key] = prediction
                
        return prediction_min_max
    

    if season == "Kharif":
        node_prev_flow = np.sum(observed_flow_values[index - 183 : index])
        #rim_station = rim_station_name
        rim_station_seasonal = rim_station_name + "_Rabi" 
        mean_value = calculate_matching_years(node_prev_flow, rim_station_seasonal)
        
        prediction = probabilities_average_flow(node_prev_flow, mean_value, season)

    else:
        node_prev_flow = np.sum(observed_flow_values[index - 183 : index])
        rim_station_seasonal = rim_station_name + "_Kharif" 
        mean_value = calculate_matching_years(node_prev_flow, rim_station_seasonal)
        prediction = probabilities_average_flow(node_prev_flow, mean_value, season)
    return prediction

def IRSA_prediction__Kharif(rim_station, input_data, season, current_index, percentage_tolerance):
    """
    Predicts the flow for a given rim station and season based on historical flow data and probability tables.
    
    Parameters:
    -----------
    rim_station : str
        The name of the rim station for which the prediction is made.
        
    input_data : pd.DataFrame
        A DataFrame containing observed flow values, historical flows, and probability tables.
        
    season : str
        The season for which to predict the flow ("Kharif" or "Rabi").
        
    current_index : int
        The index representing the current day or period in the observed flow data.
        
    percentage_tolerance : float
        The percentage range for tolerance when finding matching years for flow interpolation.
        
    Returns:
    --------
    dict
        A dictionary containing flow predictions for "Minimum" and "Maximum" probability scenarios for the given season.
    """
    
    # Extract necessary data from the input dataframe
    percentage_tolerance = percentage_tolerance[rim_station]
    observed_flows = input_data['/observed_flow'][rim_station]
    probability_table = input_data[rim_station]
    historical_flow_data = input_data['/Historical_flow_new']
    probability_intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def find_matching_years(previous_flow, seasonal_rim_station):
        """
        Finds the historical years where the flow is within a specified tolerance of the previous season flow.
        
        Parameters:
        -----------
        previous_flow : float
            The calculated flow from the previous season or time period.
            
        seasonal_rim_station : str
            The name of the rim station for the given season.
            
        Returns:
        --------
        float
            The mean flow value from the matching historical years.
        """
        lower_bound = previous_flow * (1 - percentage_tolerance)
        upper_bound = previous_flow * (1 + percentage_tolerance)
        
        matching_flow_data = historical_flow_data[(historical_flow_data[seasonal_rim_station] >= lower_bound) & 
                                                  (historical_flow_data[seasonal_rim_station] <= upper_bound)]
        
        if rim_station + "_Rabi" == seasonal_rim_station:
            mean_flow = {}
            mean_flow["Early_Kharif"] = matching_flow_data[rim_station + "_Early_Kharif"].mean()
            mean_flow["Late_Kharif"] = matching_flow_data[rim_station + "_Late_Kharif"].mean()
            
        return mean_flow
    def calculate_seasonal_predictions(mean_flow, season):
        """
        Calculates the flow predictions based on probability tables and differences from the mean flow.

        Parameters:
        -----------
        mean_flow : float or dict
            The average flow from historical data matching the previous flow.

        season : str
            The season for which the prediction is being made ("Kharif" or "Rabi").

        Returns:
        --------
        dict
            A dictionary containing flow predictions for "Minimum" and "Maximum" scenarios.
        """
        def get_probability_val(differences):
            if differences.idxmin() == "SYN Max":
                return 0.05
            elif differences.idxmin() == "SYN Min":
                return 0.95
            return differences.idxmin()

        def make_prediction(differences, period, adjustment):
            probability_val = get_probability_val(differences)
            adjusted_probability = probability_val + adjustment
            closest_probability = min(probability_intervals, key=lambda x: abs(x - adjusted_probability))
            return np.array(probability_table[closest_probability][period])

        prediction_results = {}
        min_max_scenarios = {'Minimum': -0.1, 'Maximum': 0.1}

        if season == "Kharif":
            EK_season, LK_season = probability_table.loc[:7].sum(), probability_table.loc[7:18].sum()
            differences_EK = abs(EK_season - mean_flow["Early_Kharif"])
            differences_LK = abs(LK_season - mean_flow["Late_Kharif"])

            for scenario, adjustment in min_max_scenarios.items():
                prediction_results[scenario] = {
                    "Early_Kharif": make_prediction(differences_EK, slice(0, 7), adjustment),
                    "Late_Kharif": make_prediction(differences_LK, slice(7, 18), adjustment)
                }
        else:
            end_of_kharif_season = probability_table.loc[18:36].sum()
            differences = abs(end_of_kharif_season - mean_flow)

            for scenario, adjustment in min_max_scenarios.items():
                prediction_results[scenario] = make_prediction(differences, slice(18, 36), adjustment)

        return prediction_results

    # Calculate previous season flow for the given rim station
    seasonal_flow_sum = np.sum(observed_flows[current_index - 183 : current_index])
    seasonal_rim_station = rim_station + "_Rabi" 

    # Find the matching years and calculate predictions
    average_flow = find_matching_years(seasonal_flow_sum, seasonal_rim_station)
    flow_predictions = calculate_seasonal_predictions(average_flow, season)

    return flow_predictions

class Rabi_water_allocation__(Parameter):
    def __init__(self, model, Punjab_channel_heads, Sindh_channel_heads, 
    input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, 
    Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, 
    percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, 
    Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, 
    KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, 
    System_losses_percent_Indus, System_losses_JC, **kwargs):
        
        super().__init__(model, **kwargs)
        self.input_data = input_data
        self.Mangla_reservoir = Mangla_reservoir
        self.Tarbela_reservoir = Tarbela_reservoir
        
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads
        
        self.Punjab_channel_heads_node = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}

        self.Sindh_channel_heads_Guddu = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Guddu"]}
        self.Sindh_channel_heads_Sukkur = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Sukkur"]}
        self.Sindh_channel_heads_Kotri = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Kotri"]}


        self.Sindh_channel_heads_node = {**self.Sindh_channel_heads_Guddu, **self.Sindh_channel_heads_Sukkur, **self.Sindh_channel_heads_Kotri}
        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads["Guddu"] + self.Sindh_channel_heads["Sukkur"] + self.Sindh_channel_heads["Kotri"]]

        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        
        self.Indus_at_Chashma = Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla = Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range = percentage_range
        self.Filling_withdraw_fraction_Tarbela = Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla = Filling_withdraw_fraction_Mangla
        
        self.Eastern_rivers = Eastern_rivers
        self.JC_average_system_uses_1977_1982 = JC_average_system_uses_1977_1982
        self.Average_System_use_Indus = Average_System_use_Indus
        self.KPK_Baloch_share = KPK_Baloch_share
        self.KPK_share_historical = KPK_share_historical
        self.Baloch_share_historical = Baloch_share_historical
        
        self.Below_Kotri = Below_Kotri
        self.Punjab_share_Indus_para_2_percent = Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus = System_losses_percent_Indus
        self.System_losses_JC = System_losses_JC

    def setup(self):
        self.channel_head_to_recorder_name = {
            'LJC': 'LJC_total_rec',
            'UJC_INT': 'UJC_int_rec',
            'LCC': 'LCC_rec',
            'UCC_INT': 'UCC_canal_rec',
            'LBDC': 'LBDC_before_MP_link_rec',
            'M.R_INT': 'MR_link_int_rec',

            'UDC': 'UDC_rec',
            'LDC': 'LDC_rec',
            'UPC': 'UPC_rec',
            'FC': 'fordwah_canal_rec',

            'CRBC_PB':'crbc_kpk_canal_rec', 
            'HAVELI_INT': 'havali_int_rec',
            'SIDHNAI': 'sidhnai_canal_rec', 
            'RANGPUR': 'rangpur_canal_rec',
            'PANJNAD': 'panjnad_canal_rec',
            'ABBASIA LINK': 'abbasia_link_canal_rec',

            'THAL': 'Thal Canal_rec' ,  
            'CBDC': 'CBDC_rec',
            'UBC+QC': ['qaim_canal_rec', 'UBC_rec'],
            'LBC': 'LBC_rec',
            'MZG': 'muzafargarh_canal_rec',
            'DG_KHAN': 'DGK_canal_rec'
            }
        super().setup()

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = 314166/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
        
        if start_year < ts.year:
            if ts.month == 9 and ts.day > 29:
                self.Rabi_index = 0
                self.Punjab_Sindh_channel_heads_index = 0
                
                self.Rabi_season_start_date_index = ts.index
                water_balance_J_C_zone_output = water_balance_J_C_zone(self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC) 
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"]
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]
                
                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()
            # If the current date is in the first quarter of the year, the Rabi season started last year
            if ts.month <= 3:  # January, February, March
                self.Rabi_start_time = pd.Timestamp(f'{ts.year-1}-10-01')
                self.Rabi_end_time = pd.Timestamp(f'{ts.year}-03-30')
            else:
                # Otherwise, the Rabi season is in the current year and the next year
                self.Rabi_start_time = pd.Timestamp(f'{ts.year}-10-01')
                self.Rabi_end_time = pd.Timestamp(f'{ts.year+1}-03-30')


            if self.Rabi_start_time <= ts.datetime <= self.Rabi_end_time and ts.datetime >  pd.Timestamp('2013-10-01'):
                
                #water used sofar
                self.pubjab_abstructed_Rabi = {recorder : sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder : sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node : sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Guddu[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Guddu}
                self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Sukkur[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Sukkur}
                self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_Kotri[node].max_flow.dataframe[ts.datetime : self.Rabi_end_time].values) for node in self.Sindh_channel_heads_Kotri}
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                if ts.day == 10 or ts.day == 20:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    
                    #check if the aggregated required water demand is greather aggregated available water
                    if sum(self.pubjab_remaining_demand_Rabi.values()) > self.Punjab_J_C_Channel_available_water_to_allocate + self.Punjab_Indus_Channel_available_water_to_allocate:
                        #calculate which channel head to curtail and by how much
                        Punjab_channel_heads = list(self.Punjab_J_C_Channel_dis_df_likely.columns) + list(self.Punjab_Indus_Channel_dis_df_likely.columns)
                        for channel_head in Punjab_channel_heads:
                            try:
                                whats_should_be_allocated = self.Punjab_J_C_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            except KeyError:
                                whats_should_be_allocated = self.Punjab_Indus_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            if channel_head not in  ['ESC', 'LPC', 'LMC', 'CBDC', 'UBC+QC']:
                                channel_head_node_name = self.channel_head_to_recorder_name[channel_head][:-4]
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01
                                    #self.Punjab_channel_heads_node[channel_head_node_name].max_flow = self.Punjab_channel_heads_node[channel_head_node_name].get_max_flow(scenario_index)*(1- curtailment)
                            if channel_head == 'UBC+QC':
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][0]][0] + self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][1]][0]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01 
                            
                    if sum(self.sindh_remaining_demand_Rabi.values()) > self.Sindh_Channel_available_water_to_allocate:
                        #sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index])
                        self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Guddu}
                        self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Sukkur}
                        self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Kotri}
                        
                        for channel_head in list(self.Sindh_Channel_dis_df_likely.columns):
                            whats_should_be_allocated = (self.Sindh_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum())*43560
                            if channel_head == "Guddu":
                                if sum(self.Guddu_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Guddu_remaining_demand_Rabi.values()) 
                            elif channel_head == "Sukkur":
                                if sum(self.Sukkur_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Sukkur_remaining_demand_Rabi.values()) 
                            else:
                                if sum(self.Kotri_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Kotri_remaining_demand_Rabi.values())

                    self.Rabi_index += 1
                
                elif ts.day == days_in_month:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560

                    #check if the aggregated required water demand is greather aggregated available water
                    if sum(self.pubjab_remaining_demand_Rabi.values()) > self.Punjab_J_C_Channel_available_water_to_allocate + self.Punjab_Indus_Channel_available_water_to_allocate:
                        #calculate which channel head to curtail and by how much
                        Punjab_channel_heads = list(self.Punjab_J_C_Channel_dis_df_likely.columns) + list(self.Punjab_Indus_Channel_dis_df_likely.columns)
                        for channel_head in Punjab_channel_heads:
                            try:
                                whats_should_be_allocated = self.Punjab_J_C_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            except KeyError:
                                whats_should_be_allocated = self.Punjab_Indus_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum()
                            if channel_head not in  ['ESC', 'LPC', 'LMC', 'CBDC', 'UBC+QC']:
                                channel_head_node_name = self.channel_head_to_recorder_name[channel_head][:-4]
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01
                                    #self.Punjab_channel_heads_node[channel_head_node_name].max_flow = self.Punjab_channel_heads_node[channel_head_node_name].get_max_flow(scenario_index)*(1- curtailment)
                            if channel_head == 'UBC+QC':
                                what_is_allocated = self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][0]] + self.pubjab_abstructed_Rabi[self.channel_head_to_recorder_name[channel_head][1]]
                                if what_is_allocated > whats_should_be_allocated:
                                    curtailment = 0.01 
                            
                    if sum(self.sindh_remaining_demand_Rabi.values()) > self.Sindh_Channel_available_water_to_allocate:
                        #sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index])
                        self.Guddu_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Guddu}
                        self.Sukkur_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Sukkur}
                        self.Kotri_remaining_demand_Rabi  = {node : sum(self.Sindh_channel_heads_recorders[node + "_rec"].data[self.Rabi_season_start_date_index:ts.index]) for node in self.Sindh_channel_heads_Kotri}
                        
                        for channel_head in list(self.Sindh_Channel_dis_df_likely.columns):
                            whats_should_be_allocated = (self.Sindh_Channel_dis_df_likely[channel_head][:self.Rabi_index].sum())*43560
                            if channel_head == "Guddu":
                                if sum(self.Guddu_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Guddu_remaining_demand_Rabi.values()) 
                            elif channel_head == "Sukkur":
                                if sum(self.Sukkur_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Sukkur_remaining_demand_Rabi.values()) 
                            else:
                                if sum(self.Kotri_remaining_demand_Rabi.values()) > whats_should_be_allocated:
                                    curtailment = whats_should_be_allocated/sum(self.Kotri_remaining_demand_Rabi.values())
                    self.Rabi_index += 1
                    
        else:
            self.track_year = ts.year
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")
        
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")
        
        return cls(model, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data)
Rabi_water_allocation__.register()


def Kharif_Indus_J_C_zone_water_balance(KPK_Baloch_share_Early_Kharif, KPK_Baloch_share_Late_Kharif, Below_Kotri_Early_Kharif, Below_Kotri_Late_Kharif, Indus_Early_Kharif_loss_percent, 
Indus_Late_Kharif_loss_percent, J_C_Early_Kharif_loss, 
J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, input_data,Initial_Storage_Mangla,Maximum_Storage_Mangla,
Initial_Storage_Tarbela,Maximum_Storage_Tarbela,
Indus_at_Chashma,Storage_Dep_at_end_of_season_Mangla,Storage_Dep_at_end_of_Season_Tarbela,start_date,percentage_range,
Filling_withdraw_fraction_Tarbela,Filling_withdraw_fraction_Mangla,Eastern_rivers,
JC_average_system_uses_1977_1982,Average_System_use_Indus,
KPK_Baloch_share,KPK_share_historical,Baloch_share_historical,Below_Kotri,
Punjab_share_Indus_para_2_percent,System_losses_percent_Indus,System_losses_JC):
    

    # Define the shortage percentages and ten-day index
    shortage_percentages = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ten_day_indices = np.arange(1, 19)  # Assuming consecutive numbering

    # Define the table values (you need to replace this with actual values from your dataset)
    Kharif_shortage_look_up_table = np.array([
        [34.4, 29.5, 24.0, 24.0, 24.0, 24.0, 24.0],
        [37.6, 30.4, 26.0, 26.0, 26.0, 26.0, 26.0],
        [44.5, 40.0, 33.2, 33.2, 33.2, 33.2, 33.2],
        [49.5, 49.5, 49.5, 32.9, 32.9, 32.9, 32.9],
        [52.1, 52.1, 52.1, 52.1, 35.4, 35.4, 35.4],
        [54.3, 54.3, 54.3, 54.3, 54.3, 39.0, 39.0],
        [55.7, 55.7, 55.7, 55.7, 55.7, 55.7, 39.0],
        [57.7, 57.7, 57.7, 57.7, 36.0, 30.0, 30.0],
        [59.4, 59.4, 59.4, 59.4, 59.4, 59.4, 49.0],
        [59.5, 59.5, 59.5, 59.5, 59.5, 59.5, 49.0],
        [54.5, 54.5, 54.5, 54.5, 54.5, 54.5, 49.0],
        [52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 50.0],
        [53.1, 53.1, 53.1, 53.1, 53.1, 53.1, 50.0],
        [58.7, 58.7, 58.7, 58.7, 58.7, 48.7, 48.7],
        [62.1, 62.1, 62.1, 38.0, 34.0, 30.0, 30.0],
        [61.4, 53.5, 41.0, 37.0, 34.0, 30.0, 30.0],
        [60.3, 48.0, 39.0, 37.0, 34.0, 30.0, 30.0],
        [57.2, 45.0, 34.0, 34.0, 34.0, 30.0, 30.0]
    ])
    interpolator = scipy.interpolate.RegularGridInterpolator((ten_day_indices, shortage_percentages), Kharif_shortage_look_up_table)
    def interpolate_value(shortage, ten_day):
        #print(ten_day,shortage)
        return interpolator([[ten_day, shortage]])[0]

    season = "Kharif"
    input_data = pd.HDFStore(input_data)
    observed_flow=input_data['/observed_flow']
    rim_station = ['Kabul_Noshehra', 'Indus_Tarbela', 'Jhelum_Mangla', 'Chenab_Marala']
    J_C_rim_stations_name = rim_station[2:4]
    Indus_rim_stations_name = rim_station[:2]
    observed_flow_index = observed_flow.index.get_loc(start_date)

    day_month_correction = [1, 1, 1.1, 1, 1, 1, 1, 1, 1.1, 1, 1, 1.1, 1, 1, 0.8, 1, 1, 1.1]
    CUMCS_MAC_conversion_factor = 0.01983471
    J_C_1977_88 = [34.4, 37.6, 44.5, 49.5, 52.1, 54.3, 55.7, 57.7, 59.4, 59.5, 54.5, 52.5, 53.1, 58.7, 62.1, 61.4, 60.3, 57.2]
    J_C_1977_88_MAF = [CUMCS_MAC_conversion_factor*day_month_correction[index]*J_C_1977_88[index] for index in range(len(J_C_1977_88))]
    J_C_1977_88_MAF_Early_Kharif = J_C_1977_88_MAF[:7]
    J_C_1977_88_MAF_Late_Kharif = J_C_1977_88_MAF[7:]
    ##############################################
    
    def convert_to_MAF(data):
        conversion_factor = 43560   # Convert million cubic feet to MAF
        converted_data = {}
        for station, data_dict in data.items():
            converted_data[station] = {}
            for category, sub_dict in data_dict.items():  # Minimum, Maximum
                converted_data[station][category] = {}
                for season, arr in sub_dict.items():  # Early_Kharif, Late_Kharif
                    converted_data[station][category][season] = np.array(arr) / conversion_factor
        return converted_data

    Rim_prediction = convert_to_MAF({rm : IRSA_prediction__Kharif(rm, input_data, season, observed_flow_index, percentage_range) for rm in rim_station})
    J_C_Rim_stations_prediction = {key: Rim_prediction[key] for key in J_C_rim_stations_name}
    Indus_Rim_stations_prediction = {key: Rim_prediction[key] for key in Indus_rim_stations_name}
    Filling_withdraw_fraction_Mangla_Early_Kharif = Filling_withdraw_fraction_Mangla[:7]
    Filling_withdraw_fraction_Mangla_Late_Kharif = Filling_withdraw_fraction_Mangla[7:]
    
    Filling_withdraw_fraction_Tarbela_Early_Kharif = Filling_withdraw_fraction_Tarbela[:7]
    Filling_withdraw_fraction_Tarbela_Late_Kharif = Filling_withdraw_fraction_Tarbela[7:]
    if season == "Kharif":
        J_C_Shortage = {}
        Punjab_J_C_Canal_Wdls_Outflow_RQBS = {}
        Sindh_Channel_dis_df = {}
        Punjab_J_C_Channel_dis_df = {}
        Punjab_Indus_Channel_dis_df = {}
        scenarios = ["Maximum", "Minimum"]
        kharif_season_phases = ['Early_Kharif', 'Late_Kharif']



        for scenario in scenarios:
            System_Outflow_RQBS_min_max = {}
            Jhelum_at_Mangla_Early_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[0]])
            Jhelum_at_Mangla_Late_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[1]])
            
            Chenab_at_Marala_Early_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[0]])
            Chenab_at_Marala_Late_Kharif = sum(J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[1]])
            
            Inflows_J_C_Command_Early_Kharif = Jhelum_at_Mangla_Early_Kharif + Chenab_at_Marala_Early_Kharif + sum(Eastern_rivers[0:7])
            Inflows_J_C_Command_Late_Kharif = Jhelum_at_Mangla_Late_Kharif + Chenab_at_Marala_Late_Kharif + sum(Eastern_rivers[7:])

            Storage_to_Fill_Mangla = Maximum_Storage_Mangla - Initial_Storage_Mangla 
            Storage_Fill_Mangla_Early_Kharif = Storage_to_fill_in_E_Kharif_Mangla * Storage_to_Fill_Mangla
            Storage_Fill_Mangla_Late_Kharif = Storage_to_fill_in_L_Kharif_Mangla * Storage_to_Fill_Mangla
            Storage_Dep_at_end_of_Season = 0.1 * Maximum_Storage_Mangla



            Storage_Release_Mangla = -1*(Storage_Fill_Mangla_Early_Kharif + Storage_Fill_Mangla_Late_Kharif) + Storage_Dep_at_end_of_Season
            
            System_Inflow_Early_Kharif = Inflows_J_C_Command_Early_Kharif - Storage_Fill_Mangla_Early_Kharif 
            System_Inflow_Late_Kharif = Inflows_J_C_Command_Late_Kharif - Storage_Fill_Mangla_Late_Kharif + Storage_Dep_at_end_of_Season
            System_Inflow = System_Inflow_Early_Kharif + System_Inflow_Late_Kharif
            Total_Availability_JC_Early_Kharif = System_Inflow_Early_Kharif * (1 - J_C_Early_Kharif_loss)
            Total_Availability_JC_Late_Kharif = System_Inflow_Late_Kharif * (1 - J_C_Late_Kharif_loss)
            Total_Availability_JC = Total_Availability_JC_Early_Kharif + Total_Availability_JC_Late_Kharif
            
            
            Indus_at_Tarbela_Early_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[0]])
            Indus_at_Tarbela_Late_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[1]])
            
            Kabul_at_Nowshehra_Early_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[0]])
            Kabul_at_Nowshehra_Late_Kharif = sum(Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[1]])
             
            ##############################IIINNNN#################################################
            Total_Indus_Early_Kharif = Indus_at_Tarbela_Early_Kharif + Kabul_at_Nowshehra_Early_Kharif
            Total_Indus_Late_Kharif = Indus_at_Tarbela_Late_Kharif + Kabul_at_Nowshehra_Late_Kharif 
            
            #print(scenario, "********************", Indus_at_Tarbela_Early_Kharif, Kabul_at_Nowshehra_Early_Kharif)
            Storage_to_Fill_Tarbela = Maximum_Storage_Tarbela - Initial_Storage_Tarbela 
            Storage_Fill_Tarbela_Early_Kharif = Storage_to_fill_in_E_Kharif_Tarbela * Storage_to_Fill_Tarbela
            Storage_Fill_Tarbela_Late_Kharif = Storage_to_fill_in_L_Kharif_Tarbela * Storage_to_Fill_Tarbela

            Storage_Dep_at_end_of_Season_Tarbela_percent = 0.1
            Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela_percent*Maximum_Storage_Tarbela
            Storage_Release_Tarbela = -1*(Maximum_Storage_Tarbela*(1 - Storage_Dep_at_end_of_Season_Tarbela_percent) - Initial_Storage_Tarbela)
            ############################################################################################################### 
            JC_average_system_uses_1977_1982_Early_Kharif = JC_average_system_uses_1977_1982["Early_Kharif"]
            JC_average_system_uses_1977_1982_Late_Kharif = JC_average_system_uses_1977_1982["Late_Kharif"]
            
            Average_System_use_Indus_Early_Kharif = Average_System_use_Indus["Early_Kharif"]
            Average_System_use_Indus_Late_Kharif = Average_System_use_Indus["Late_Kharif"]
             
            Indus_Early_Kharif_loss = 1 - Indus_Early_Kharif_loss_percent
            Indus_Late_Kharif_loss = 1 - Indus_Late_Kharif_loss_percent
     
            Baloch_share_Early_Kharif = [day_month_correction[:7][index]*Baloch_share_historical[:7][index]*0.01983471 for index in range(len(KPK_share_historical[:7]))]
            KPK_share_Early_Kharif = [day_month_correction[:7][index]*KPK_share_historical[:7][index]*0.01983471 for index in range(len(KPK_share_historical[:7]))]
            
            Baloch_share_Late_Kharif = [day_month_correction[7:][index]*Baloch_share_historical[7:][index]*0.01983471 for index in range(len(KPK_share_historical[7:]))]
            KPK_share_Late_Kharif = [day_month_correction[7:][index]*KPK_share_historical[7:][index]*0.01983471 for index in range(len(KPK_share_historical[7:]))] 
            ###############################################################################################################  

            #################################Check shortage values################################################# 
            Indus_know_parameters_Early_Kharif = Indus_Early_Kharif_loss*(Storage_Release_Tarbela + Total_Indus_Early_Kharif) - KPK_Baloch_share_Early_Kharif - Below_Kotri_Early_Kharif
            Numerator_Early_Kharif = Total_Availability_JC_Early_Kharif*Average_System_use_Indus_Early_Kharif - JC_average_system_uses_1977_1982_Early_Kharif*Indus_know_parameters_Early_Kharif
            Denominator_Early_Kharif = Indus_Early_Kharif_loss*JC_average_system_uses_1977_1982_Early_Kharif + Average_System_use_Indus_Early_Kharif
            J_C_Outflow_Early_Kharif = Numerator_Early_Kharif/Denominator_Early_Kharif

        
            Indus_know_parameters_Late_Kharif = Indus_Late_Kharif_loss*(Storage_Release_Tarbela + Total_Indus_Late_Kharif) - KPK_Baloch_share_Late_Kharif - Below_Kotri_Late_Kharif
            Numerator_Late_Kharif = Total_Availability_JC_Late_Kharif*Average_System_use_Indus_Late_Kharif - JC_average_system_uses_1977_1982_Late_Kharif*Indus_know_parameters_Late_Kharif
            Denominator_Late_Kharif = Indus_Late_Kharif_loss*JC_average_system_uses_1977_1982_Late_Kharif + Average_System_use_Indus_Late_Kharif
            J_C_Outflow_Late_Kharif = Numerator_Late_Kharif/Denominator_Late_Kharif

            ###################################Check############################################################### 
            JC_Canal_Availability_Early_Kharif = Total_Availability_JC_Early_Kharif - J_C_Outflow_Early_Kharif
            #print(scenario, J_C_Outflow_Early_Kharif, "££££££££££££££££££££££££££££",Total_Availability_JC_Early_Kharif, JC_average_system_uses_1977_1982_Early_Kharif)
            J_C_Shortage_Early_Kharif = (1 - JC_Canal_Availability_Early_Kharif/JC_average_system_uses_1977_1982_Early_Kharif)/100
            
            JC_Canal_Availability_Late_Kharif = Total_Availability_JC_Late_Kharif - J_C_Outflow_Late_Kharif
            J_C_Shortage_Late_Kharif = (1 - JC_Canal_Availability_Late_Kharif/JC_average_system_uses_1977_1982_Late_Kharif)/100
            ###################################Check###############################################################  
            

            ############################Early_Khrif################################################
            #need to extract Initial_Storage_Mangla from the Mangla storage node@@@@@@@@@@@@@
            Mangla_Live_content_MAF_Early_Kharif = [Initial_Storage_Mangla]  #in MAC
            
            Initial_Storage_temp = Initial_Storage_Mangla
            
            for vol in Filling_withdraw_fraction_Mangla_Early_Kharif:
                release = (Storage_Fill_Mangla_Early_Kharif*vol/100)
                Initial_Storage_temp = release + Initial_Storage_temp 
                Mangla_Live_content_MAF_Early_Kharif.append(Initial_Storage_temp)

            Mangla_outflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Mangla_Early_Kharif)):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[0]][index] - (Mangla_Live_content_MAF_Early_Kharif[index+1] - Mangla_Live_content_MAF_Early_Kharif[index])
                Mangla_outflow_Early_Kharif.append(release)
            #need to combine the Early and Late kharif rim station predictions
            System_Inflow_Early_Kharif = [Mangla_outflow_Early_Kharif[index] + Eastern_rivers[0:7][index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[0]][index] for index in range(len(Mangla_outflow_Early_Kharif))]
            Net_inflow_after_loss_Early_Kharif = [-1*inflow*(J_C_Early_Kharif_loss) for inflow in System_Inflow_Early_Kharif]
            Net_inflow_Early_Kharif = [System_Inflow_Early_Kharif[index] + Net_inflow_after_loss_Early_Kharif[index] for index in range(len(Net_inflow_after_loss_Early_Kharif))]
            
            
            ############################Early_Khrif################################################
            Punjab_J_C_Canal_Wdls_Early_Kharif = [
                Net_inflow_Early_Kharif[index] if interpolate_value(J_C_Shortage_Early_Kharif, index + 1) > Net_inflow_Early_Kharif[index] 
                else interpolate_value(J_C_Shortage_Early_Kharif, index + 1)
                for index in range(len(Net_inflow_Early_Kharif))
            ]
            System_Outflow_RQBS_Early_Kharif = [Net_inflow_Early_Kharif[index] - Punjab_J_C_Canal_Wdls_Early_Kharif[index] for index in range(len(Punjab_J_C_Canal_Wdls_Early_Kharif))]
            ############################Early_Khrif################################################

            #need to extract Initial_Storage_Mangla from the Mangla storage node@@@@@@@@@@@@@
            Mangla_Live_content_MAF_Late_Kharif = [Mangla_Live_content_MAF_Early_Kharif[-1]]  
            Initial_Storage_temp = Mangla_Live_content_MAF_Late_Kharif[0]

            for index in range(len(Filling_withdraw_fraction_Mangla_Late_Kharif)):
                if index > 7:
                    if index == 8:                    
                        Storage_Dep_at_end_of_season = Initial_Storage_Mangla * Storage_Dep_at_end_of_season_Mangla
                        Initial_Storage_temp = Mangla_Live_content_MAF_Early_Kharif[-1]
                        Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp - Storage_Dep_at_end_of_season)
                    else:
                        Storage_Dep_at_end_of_season = Initial_Storage_Mangla*Storage_Dep_at_end_of_season_Mangla
                        Initial_Storage_temp = Mangla_Live_content_MAF_Late_Kharif[index-1]
                        Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp - Storage_Dep_at_end_of_season)
                else:
                    vol = Filling_withdraw_fraction_Mangla_Late_Kharif[index]
                    release = (Storage_Fill_Mangla_Late_Kharif*vol/100)
                    Initial_Storage_temp = release + Initial_Storage_temp 
                    Mangla_Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)

            Mangla_outflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Mangla_Late_Kharif)):
                release = J_C_Rim_stations_prediction[J_C_rim_stations_name[0]][scenario][kharif_season_phases[1]][index] - (Mangla_Live_content_MAF_Late_Kharif[index+1] - Mangla_Live_content_MAF_Late_Kharif[index])
                Mangla_outflow_Late_Kharif.append(release)            
            #need to combine the Early and Late kharif rim station predictions
            
            System_Inflow_Late_Kharif = [Mangla_outflow_Late_Kharif[index] + Eastern_rivers[7:][index] + J_C_Rim_stations_prediction[J_C_rim_stations_name[1]][scenario][kharif_season_phases[1]][index] for index in range(len(Mangla_outflow_Late_Kharif))]
            Net_inflow_after_loss_Late_Kharif = [-1*inflow*(J_C_Late_Kharif_loss/100) for inflow in System_Inflow_Late_Kharif]
            Net_inflow_Late_Kharif = [System_Inflow_Late_Kharif[index] + Net_inflow_after_loss_Late_Kharif[index] for index in range(len(Net_inflow_after_loss_Late_Kharif))]

            Punjab_J_C_Canal_Wdls_Late_Kharif = []
            Punjab_J_C_Canal_Wdls_Late_Kharif = [
                Net_inflow_Late_Kharif[index] if (1 - J_C_Shortage_Late_Kharif * 0.01) * J_C_1977_88_MAF_Late_Kharif[index] > Net_inflow_Late_Kharif[index] 
                else (1 - J_C_Shortage_Late_Kharif * 0.01) * J_C_1977_88_MAF_Late_Kharif[index]
                for index in range(len(Net_inflow_Late_Kharif))
            ]
            System_Outflow_RQBS_Late_Kharif = [Net_inflow_Late_Kharif[index] - Punjab_J_C_Canal_Wdls_Late_Kharif[index] for index in range(len(Punjab_J_C_Canal_Wdls_Late_Kharif))]
            ############################Early_Khrif################################################

            ############################JJJJCCC################################################
            #JC_outflow is part of the below equation that need to be fixed
            
            Total_System_Inflows_Indus_Early_Kharif = Total_Indus_Early_Kharif + sum(System_Outflow_RQBS_Early_Kharif) - Storage_Fill_Tarbela_Early_Kharif
            Total_Availability_Indus_Early_Kharif = Total_System_Inflows_Indus_Early_Kharif - Indus_Early_Kharif_loss_percent*Total_System_Inflows_Indus_Early_Kharif
            
            
            #Implement paraII conditional statment
            Canal_Availability_Indus_Early_Kharif = Total_Availability_Indus_Early_Kharif - Below_Kotri_Early_Kharif

            Punjab_Sindh_share_Indus_Early_Kharif = Canal_Availability_Indus_Early_Kharif - KPK_Baloch_share_Early_Kharif
            Indus_Shortage_Early_Kharif = (1 - Punjab_Sindh_share_Indus_Early_Kharif/Average_System_use_Indus_Early_Kharif)*100

            Total_System_Inflows_Indus_Late_Kharif = Total_Indus_Late_Kharif + sum(System_Outflow_RQBS_Late_Kharif) - Storage_Fill_Tarbela_Late_Kharif
            Total_Availability_Indus_Late_Kharif = Total_System_Inflows_Indus_Late_Kharif - Indus_Late_Kharif_loss_percent*Total_System_Inflows_Indus_Late_Kharif
            Canal_Availability_Indus_Late_Kharif = Total_Availability_Indus_Late_Kharif - Below_Kotri_Late_Kharif
            Punjab_Sindh_share_Indus_Late_Kharif = Canal_Availability_Indus_Late_Kharif - KPK_Baloch_share_Late_Kharif
            Indus_Shortage_Late_Kharif = (1 - Punjab_Sindh_share_Indus_Late_Kharif/Average_System_use_Indus_Late_Kharif)*100

            ##############################IIINNNN#################################################
            ##############################IIINNNN##################################################
            #need to extract Initial_Storage_Tarbela from the Mangla storage node@@@@@@@@@@@@@
            
            Filling_withdraw_fraction_Tarbela_Early_Kharif_1 = Filling_withdraw_fraction_Tarbela[:7]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_2 = Filling_withdraw_fraction_Tarbela[7:9]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_3 = [4.529, 5.054, 5.602, Maximum_Storage_Tarbela, Maximum_Storage_Tarbela, Maximum_Storage_Tarbela]
            Filling_withdraw_fraction_Tarbela_Early_Kharif_4 = Filling_withdraw_fraction_Tarbela[15:18]


            Live_content_MAF_Early_Kharif = [Initial_Storage_Tarbela]  #in MAC
            Storage_Fill_Tarbela_Early_Kharif = 0.23*(Maximum_Storage_Tarbela - Initial_Storage_Tarbela) 
            Storage_Fill_Tarbela_Late_Kharif = 0.77*(Maximum_Storage_Tarbela - Initial_Storage_Tarbela) 

            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_1:
                release = (Storage_Fill_Tarbela_Early_Kharif*vol/100)
                Initial_Storage_temp = release + Initial_Storage_temp 
                Live_content_MAF_Early_Kharif.append(Initial_Storage_temp)

            Live_content_MAF_Late_Kharif = [Live_content_MAF_Early_Kharif[-1]]
            Tarbela_filling_Limit_start = 4.025
            Live_content_MAF_constat = Live_content_MAF_Early_Kharif[-1]
            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_2:
                release = ((Tarbela_filling_Limit_start - Live_content_MAF_constat)*vol/100) 
                Initial_Storage_temp = release + Initial_Storage_temp 
                Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)


            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_3:
                Live_content_MAF_Late_Kharif.append(vol)
            
            Storage_Dep_at_end_of_season = 0.1
            Tarbela_Max_Storage = 6.17
            Initial_Storage_temp = Live_content_MAF_Late_Kharif[-1]
            for vol in Filling_withdraw_fraction_Tarbela_Early_Kharif_4:
                release = (Initial_Storage_temp - Storage_Dep_at_end_of_season*Tarbela_Max_Storage*vol/100) 
                Initial_Storage_temp = release  
                Live_content_MAF_Late_Kharif.append(Initial_Storage_temp)


            Tarbela_outflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Early_Kharif)):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[0]][index]-(Live_content_MAF_Early_Kharif[index+1]-Live_content_MAF_Early_Kharif[index])
                Tarbela_outflow_Early_Kharif.append(release)

            Tarbela_outflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Late_Kharif)):
                release = Indus_Rim_stations_prediction[Indus_rim_stations_name[1]][scenario][kharif_season_phases[1]][index]-(Live_content_MAF_Late_Kharif[index+1]-Live_content_MAF_Late_Kharif[index])
                Tarbela_outflow_Late_Kharif.append(release)

            ParaII = [68.2, 70.1, 79.1, 99.6, 116.5, 135.3, 162.8, 187.2, 198.7, 205.0, 186.2, 175.9, 169.6, 168.6, 175.7, 177.3, 175.1, 170.1]
            ParaII_Early_Kharif = ParaII[:7] 
            ParaII_Late_Kharif = ParaII[7:] 
            Initial_Storage_temp = Initial_Storage_Tarbela
            Chashma_storage_Early_Kharif = [2.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            Chashma_storage_Late_Kharif = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, -2.0, 2.0, -5.0]

            Indus_Net_System_Inflow_Early_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Early_Kharif)):
                Inflow = (Tarbela_outflow_Early_Kharif[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[0]][index] + Chashma_storage_Early_Kharif[index] + System_Outflow_RQBS_Early_Kharif[index])*Indus_Early_Kharif_loss
                Indus_Net_System_Inflow_Early_Kharif.append(Inflow)

            Indus_Net_System_Inflow_Late_Kharif = []
            for index in range(len(Filling_withdraw_fraction_Tarbela_Late_Kharif)):
                Inflow = (Tarbela_outflow_Late_Kharif[index] + Indus_Rim_stations_prediction[Indus_rim_stations_name[0]][scenario][kharif_season_phases[1]][index] + Chashma_storage_Late_Kharif[index] + System_Outflow_RQBS_Late_Kharif[index])*Indus_Late_Kharif_loss
                Indus_Net_System_Inflow_Late_Kharif.append(Inflow)
            
            Proposed_Canal_Wdls_Indus_Early_Kharif = [ParaII_Early_Kharif[index] if Indus_Net_System_Inflow_Early_Kharif[index]>ParaII_Early_Kharif[index] else Indus_Net_System_Inflow_Early_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Early_Kharif))]     
            System_Outflow_DS_Kotri_Early_Kharif = [Indus_Net_System_Inflow_Early_Kharif[index] - Proposed_Canal_Wdls_Indus_Early_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Early_Kharif))]
            Total_Share_Punjab_Sindh_Early_Kharif = [Proposed_Canal_Wdls_Indus_Early_Kharif[index] - KPK_share_Early_Kharif[index] - Baloch_share_Early_Kharif[index] for index in range(len(KPK_share_Early_Kharif))]

            Proposed_Canal_Wdls_Indus_Late_Kharif = [ParaII_Late_Kharif[index] if Indus_Net_System_Inflow_Late_Kharif[index]>ParaII_Late_Kharif[index] else Indus_Net_System_Inflow_Late_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Late_Kharif))]     
            System_Outflow_DS_Kotri_Late_Kharif = [Indus_Net_System_Inflow_Late_Kharif[index] - Proposed_Canal_Wdls_Indus_Late_Kharif[index] for index in range(len(Indus_Net_System_Inflow_Late_Kharif))]
            Total_Share_Punjab_Sindh_Late_Kharif = [Proposed_Canal_Wdls_Indus_Late_Kharif[index] - KPK_share_Late_Kharif[index] - Baloch_share_Late_Kharif[index] for index in range(len(KPK_share_Late_Kharif))]
        
            Total_Share_Punjab_Sindh = Total_Share_Punjab_Sindh_Early_Kharif + Total_Share_Punjab_Sindh_Late_Kharif 

            #condition to implement the Para2 para14b and 
            Sindh_share_Indus_para_2_percent = [100 - Punjab_share_Indus_para_2_percent[index] for index in range(len(Punjab_share_Indus_para_2_percent))]
            Punjab_Indus_Canal_Wdls = [Total_Share_Punjab_Sindh[index]*Punjab_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]
            Sindh_share_Canal_Wdls  = [Total_Share_Punjab_Sindh[index]*Sindh_share_Indus_para_2_percent[index]*0.01 for index in range(len(Punjab_share_Indus_para_2_percent))]

            ##############################IIINNNN#######################################################################
            Punjab_J_C_Canal_Wdls = Punjab_J_C_Canal_Wdls_Early_Kharif + Punjab_J_C_Canal_Wdls_Late_Kharif 

            Sindh_Channel_dis_df[scenario] = input_data['Sindh_Channel_dis_plan_percentage'][18:].mul(Sindh_share_Canal_Wdls, axis=0)
            Punjab_J_C_Channel_dis_df[scenario] = input_data['J_C_Channel_dis_plan_percentage'][18:].mul(Punjab_J_C_Canal_Wdls, axis=0)
            Punjab_Indus_Channel_dis_df[scenario] = input_data['Indus_Channel_dis_plan_percentage'][18:].mul(Punjab_Indus_Canal_Wdls, axis=0)
            System_Outflow_RQBS_min_max[scenario] = System_Outflow_RQBS_Late_Kharif + System_Outflow_RQBS_Late_Kharif
                
        output = {}
        Sindh_Channel_dis_df_likely = (Sindh_Channel_dis_df[scenarios[0]] + Sindh_Channel_dis_df[scenarios[1]]) / 2
        Punjab_J_C_Channel_dis_df_likely = (Punjab_J_C_Channel_dis_df[scenarios[0]] + Punjab_J_C_Channel_dis_df[scenarios[1]]) / 2
        Punjab_Indus_Channel_dis_df_likely = (Punjab_Indus_Channel_dis_df[scenarios[0]] + Punjab_Indus_Channel_dis_df[scenarios[1]]) / 2
        RQBS_Canal_Outflow_likely =  [sum(x) / len(x) for x in zip(*System_Outflow_RQBS_min_max.values())]
        
        output["Sindh_Channel_dis_df_likely"] = Sindh_Channel_dis_df_likely
        output["Punjab_J_C_Channel_dis_df_likely"] = Punjab_J_C_Channel_dis_df_likely
        output["Punjab_Indus_Channel_dis_df_likely"] = Punjab_Indus_Channel_dis_df_likely
        output["RQBS_Canal_Outflow_likely"] = RQBS_Canal_Outflow_likely
  
    return output


class Kharif_water_allocation(Parameter):
    def __init__(self, model,KPK_Baloch_share_Early_Kharif,KPK_Baloch_share_Late_Kharif,Below_Kotri_Early_Kharif,
    Below_Kotri_Late_Kharif,Indus_Early_Kharif_loss_percent,Indus_Late_Kharif_loss_percent,
    J_C_Early_Kharif_loss, J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, 
    Punjab_channel_heads, Sindh_channel_heads, 
    input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, 
    Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, 
    percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, 
    Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, 
    KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, 
    System_losses_percent_Indus, System_losses_JC, **kwargs):
        
        super().__init__(model, **kwargs)


        self.KPK_Baloch_share_Early_Kharif = KPK_Baloch_share_Early_Kharif
        self.KPK_Baloch_share_Late_Kharif = KPK_Baloch_share_Late_Kharif
        self.Below_Kotri_Early_Kharif = Below_Kotri_Early_Kharif
        self.Below_Kotri_Late_Kharif = Below_Kotri_Late_Kharif
        self.Indus_Early_Kharif_loss_percent = Indus_Early_Kharif_loss_percent
        self.Indus_Late_Kharif_loss_percent = Indus_Late_Kharif_loss_percent


        self.J_C_Early_Kharif_loss = J_C_Early_Kharif_loss
        self.J_C_Late_Kharif_loss = J_C_Late_Kharif_loss
        self.input_data = input_data
        self.Mangla_reservoir = Mangla_reservoir
        self.Tarbela_reservoir = Tarbela_reservoir
        
        self.Punjab_channel_heads = Punjab_channel_heads
        self.Sindh_channel_heads = Sindh_channel_heads

        self.Storage_to_fill_in_E_Kharif_Mangla = Storage_to_fill_in_E_Kharif_Mangla
        self.Storage_to_fill_in_L_Kharif_Mangla = Storage_to_fill_in_L_Kharif_Mangla
        self.Storage_to_fill_in_E_Kharif_Tarbela = Storage_to_fill_in_E_Kharif_Tarbela
        self.Storage_to_fill_in_L_Kharif_Tarbela = Storage_to_fill_in_L_Kharif_Tarbela
        self.Sindh_channel_heads_Guddu = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Guddu"]}
        self.Sindh_channel_heads_Sukkur = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Sukkur"]}
        self.Sindh_channel_heads_Kotri = {node_name : model._get_node_from_ref(model, node_name) for node_name in self.Sindh_channel_heads["Kotri"]}

        self.Sindh_channel_heads_node = {**self.Sindh_channel_heads_Guddu, **self.Sindh_channel_heads_Sukkur, **self.Sindh_channel_heads_Kotri}
        self.Punjab_channel_heads_node = {node_name:model._get_node_from_ref(model, node_name) for node_name in self.Punjab_channel_heads}
        

        self.Punjab_channel_heads_recorders_name = [node+"_rec" for node in self.Punjab_channel_heads]
        self.Sindh_channel_heads_recorders_name = [node+"_rec" for node in self.Sindh_channel_heads["Guddu"] + self.Sindh_channel_heads["Sukkur"] + self.Sindh_channel_heads["Kotri"]]
        self.Punjab_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Punjab_channel_heads_recorders_name}
        self.Sindh_channel_heads_recorders = {rec_name:load_recorder(model, rec_name) for rec_name in self.Sindh_channel_heads_recorders_name}
        

        self.Indus_at_Chashma = Indus_at_Chashma
        self.Storage_Dep_at_end_of_season_Mangla = Storage_Dep_at_end_of_season_Mangla
        self.Storage_Dep_at_end_of_Season_Tarbela = Storage_Dep_at_end_of_Season_Tarbela
        self.percentage_range = percentage_range
        self.Filling_withdraw_fraction_Tarbela = Filling_withdraw_fraction_Tarbela
        self.Filling_withdraw_fraction_Mangla = Filling_withdraw_fraction_Mangla
        
        self.Eastern_rivers = Eastern_rivers
        self.JC_average_system_uses_1977_1982 = JC_average_system_uses_1977_1982
        self.Average_System_use_Indus = Average_System_use_Indus
        self.KPK_Baloch_share = KPK_Baloch_share
        self.KPK_share_historical = KPK_share_historical
        self.Baloch_share_historical = Baloch_share_historical
        
        self.Below_Kotri = Below_Kotri
        self.Punjab_share_Indus_para_2_percent = Punjab_share_Indus_para_2_percent
        self.System_losses_percent_Indus = System_losses_percent_Indus
        self.System_losses_JC = System_losses_JC

    def setup(self):
        super().setup()

    def value(self, timestep, scenario_index):
        i = scenario_index.global_id
        ts = self.model.timestepper.current
        days_in_month = timestep.period.days_in_month
        start_date = str(ts.year)+"-"+str(ts.month)+"-"+str(ts.day)
        self.val = 0
        start_year = self.model.timestepper.start.year
        Initial_Storage_Mangla = self.Mangla_reservoir.volume[i]/43560
        Maximum_Storage_Mangla = self.Mangla_reservoir.max_volume/43560
        Initial_Storage_Tarbela = self.Tarbela_reservoir.volume[i]/43560
        Maximum_Storage_Tarbela = 314166/43560
        
        self.pubjab_abstructed_Rabi = {}
        self.sindh_abstructed_Rabi = {}
        
        self.pubjab_remaining_demand_Rabi = {}
        self.sindh_remaining_demand_Rabi = {}
        
        if ts.month == 9 and ts.day > 29:
            self.Rabi_season_start_date_index = ts.index
            
        if start_year < ts.year:
            if ts.month == 3 and ts.day > 30:
                self.Rabi_index = 0
                self.Punjab_Sindh_channel_heads_index = 0
                self.Rabi_season_start_date_index = ts.index

                water_balance_J_C_zone_output = Kharif_Indus_J_C_zone_water_balance(self.KPK_Baloch_share_Early_Kharif,
                    self.KPK_Baloch_share_Late_Kharif,
                    self.Below_Kotri_Early_Kharif,
                    self.Below_Kotri_Late_Kharif,
                    self.Indus_Early_Kharif_loss_percent,
                    self.Indus_Late_Kharif_loss_percent,
                    
                    self.J_C_Early_Kharif_loss, self.J_C_Late_Kharif_loss, self.Storage_to_fill_in_E_Kharif_Tarbela, self.Storage_to_fill_in_L_Kharif_Tarbela, self.Storage_to_fill_in_E_Kharif_Mangla, self.Storage_to_fill_in_L_Kharif_Mangla, self.input_data, Initial_Storage_Mangla, Maximum_Storage_Mangla, Initial_Storage_Tarbela, Maximum_Storage_Tarbela, self.Indus_at_Chashma, self.Storage_Dep_at_end_of_season_Mangla, self.Storage_Dep_at_end_of_Season_Tarbela, start_date, self.percentage_range, self.Filling_withdraw_fraction_Tarbela, self.Filling_withdraw_fraction_Mangla, self.Eastern_rivers, self.JC_average_system_uses_1977_1982, self.Average_System_use_Indus, self.KPK_Baloch_share, self.KPK_share_historical, self.Baloch_share_historical, self.Below_Kotri, self.Punjab_share_Indus_para_2_percent, self.System_losses_percent_Indus, self.System_losses_JC)
                
                
                self.Sindh_Channel_dis_df_likely = water_balance_J_C_zone_output["Sindh_Channel_dis_df_likely"] 
                self.Punjab_J_C_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_J_C_Channel_dis_df_likely"]
                self.Punjab_Indus_Channel_dis_df_likely = water_balance_J_C_zone_output["Punjab_Indus_Channel_dis_df_likely"]
                self.RQBS_Canal_Outflow_likely = water_balance_J_C_zone_output["RQBS_Canal_Outflow_likely"]
                
                self.Sindh_total_allocated_water = self.Sindh_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_J_C_total_allocated_water = self.Punjab_J_C_Channel_dis_df_likely.sum(axis=1).tolist()
                self.Punjab_Indus_total_allocated_water = self.Punjab_Indus_Channel_dis_df_likely.sum(axis=1).tolist()
            
            Rabi_start_time = pd.to_datetime(str(ts.year) + '-09-29')
            Rabi_end_time = pd.to_datetime(str(ts.year + 1) + '-03-30') 
            if Rabi_start_time <= ts.datetime <= Rabi_end_time:
               
                
                #water used sofar
                self.pubjab_abstructed_Rabi = {recorder:sum(self.Punjab_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Punjab_channel_heads_recorders}
                self.sindh_abstructed_Rabi = {recorder:sum(self.Sindh_channel_heads_recorders[recorder].data[self.Rabi_season_start_date_index:ts.index]) for recorder in self.Sindh_channel_heads_recorders}
                
                #remining water that is required to satisfy the full demand
                self.pubjab_remaining_demand_Rabi = {node:sum(self.Punjab_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Punjab_channel_heads_node}
                self.sindh_remaining_demand_Rabi  = {node:sum(self.Sindh_channel_heads_node[node].max_flow.dataframe[ts.datetime : Rabi_end_time].values) for node in self.Sindh_channel_heads_node}
                
                
                #how much avaiable is avaiable to allocate according to IRSA's prediction 
                if ts.day == 10 or ts.day == 20:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
         
                    self.Rabi_index += 1
                    
                elif ts.day == days_in_month:
                    self.Sindh_Channel_available_water_to_allocate = sum(self.Sindh_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_J_C_Channel_available_water_to_allocate = sum(self.Punjab_J_C_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    self.Punjab_Indus_Channel_available_water_to_allocate = sum(self.Punjab_Indus_Channel_dis_df_likely[self.Rabi_index:].sum())*43560
                    
                    self.Rabi_index += 1
        
        #need the list of nodes in Punjab J-C and Indus Zone and Sindh Zone
        #check the next 183 days demand and if there less water reduce the demand by a certain fraction for those who are using more than the allocated amount
        #J-C outflow need to be tracked everytime and the value need to match with the allocated one
        
        #step1: collect all the recorders to get how much water is allocated 
        #step2: then create a number of 
        
        
        val = sum(self.pubjab_remaining_demand_Rabi.values())
        return val
            
    @classmethod
    def load(cls, model, data):
        Mangla_reservoir = model._get_node_from_ref(model, data.pop("Mangla_reservoir_node"))
        Tarbela_reservoir = model._get_node_from_ref(model, data.pop("Tarbela_reservoir_node"))
        
        Indus_at_Chashma = data.pop("Indus_at_Chashma")
        Storage_Dep_at_end_of_season_Mangla = data.pop("Storage_Dep_at_end_of_season_Mangla")
        Storage_Dep_at_end_of_Season_Tarbela = data.pop("Storage_Dep_at_end_of_Season_Tarbela")
        System_losses = data.pop("System_losses")
        
        percentage_range = data.pop("percentage_range")
        Filling_withdraw_fraction_Tarbela = data.pop("Filling_withdraw_fraction_Tarbela")
        Filling_withdraw_fraction_Mangla = data.pop("Filling_withdraw_fraction_Mangla")
        Eastern_rivers = data.pop("Eastern_rivers")
        
        JC_average_system_uses_1977_1982 = data.pop("JC_average_system_uses_1977_1982")
        Average_System_use_Indus = data.pop("Average_System_use_Indus")
        KPK_Baloch_share = data.pop("KPK_Baloch_share")
        KPK_share_historical = data.pop("KPK_share_historical")
        
        Baloch_share_historical = data.pop("Baloch_share_historical")
        Below_Kotri = data.pop("Below_Kotri")
        Punjab_share_Indus_para_2_percent = data.pop("Punjab_share_Indus_para_2_percent")
        System_losses_percent_Indus = data.pop("System_losses_percent_Indus")
        System_losses_JC = data.pop("System_losses_JC")

        Storage_to_fill_in_E_Kharif_Tarbela = data.pop("Storage_to_fill_in_E_Kharif_Tarbela")
        Storage_to_fill_in_L_Kharif_Tarbela = data.pop("Storage_to_fill_in_L_Kharif_Tarbela")

        Storage_to_fill_in_E_Kharif_Mangla = data.pop("Storage_to_fill_in_E_Kharif_Mangla")
        Storage_to_fill_in_L_Kharif_Mangla = data.pop("Storage_to_fill_in_L_Kharif_Mangla")
        input_data = data.pop("url")
        Punjab_channel_heads = data.pop("Punjab_channel_heads")
        Sindh_channel_heads = data.pop("Sindh_channel_heads")

        J_C_Early_Kharif_loss = data.pop("J_C_Early_Kharif_loss")
        J_C_Late_Kharif_loss = data.pop("J_C_Late_Kharif_loss")

        KPK_Baloch_share_Early_Kharif = data.pop("KPK_Baloch_share_Early_Kharif")
        KPK_Baloch_share_Late_Kharif = data.pop("KPK_Baloch_share_Late_Kharif")
        Below_Kotri_Early_Kharif = data.pop("Below_Kotri_Early_Kharif")
        Below_Kotri_Late_Kharif = data.pop("Below_Kotri_Late_Kharif")
        Indus_Early_Kharif_loss_percent = data.pop("Indus_Early_Kharif_loss_percent")
        Indus_Late_Kharif_loss_percent = data.pop("Indus_Late_Kharif_loss_percent")
        return cls(model, KPK_Baloch_share_Early_Kharif, KPK_Baloch_share_Late_Kharif, Below_Kotri_Early_Kharif, Below_Kotri_Late_Kharif, Indus_Early_Kharif_loss_percent, Indus_Late_Kharif_loss_percent, J_C_Early_Kharif_loss, J_C_Late_Kharif_loss, Storage_to_fill_in_E_Kharif_Tarbela, Storage_to_fill_in_L_Kharif_Tarbela, Storage_to_fill_in_E_Kharif_Mangla, Storage_to_fill_in_L_Kharif_Mangla, Punjab_channel_heads, Sindh_channel_heads, input_data, Mangla_reservoir, Tarbela_reservoir, Indus_at_Chashma, Storage_Dep_at_end_of_season_Mangla, Storage_Dep_at_end_of_Season_Tarbela, percentage_range, Filling_withdraw_fraction_Tarbela, Filling_withdraw_fraction_Mangla, Eastern_rivers, JC_average_system_uses_1977_1982, Average_System_use_Indus, KPK_Baloch_share, KPK_share_historical, Baloch_share_historical, Below_Kotri, Punjab_share_Indus_para_2_percent, System_losses_percent_Indus, System_losses_JC, **data) 
Kharif_water_allocation.register()

class CSVRecorder(Recorder):
    """
    A Recorder that saves Node values to a CSV file.

    This class uses the csv package from the Python standard library

    Parameters
    ----------

    model : `pywr.model.Model`
        The model to record nodes from.
    csvfile : str
        The path to the CSV file.
    scenario_index : int
        The scenario index of the model to save.
    nodes : iterable (default=None)
        An iterable of nodes to save data. It defaults to None which is all nodes in the model
    kwargs : Additional keyword arguments to pass to the `csv.writer` object

    """
    def __init__(self, model, csvfile, scenario_index=0, nodes=None, complib=None, complevel=9, **kwargs):
        super(CSVRecorder, self).__init__(model, **kwargs)
        self.csvfile = csvfile
        self.scenario_index = scenario_index
        self.nodes = nodes
        self.csv_kwargs = kwargs.pop('csv_kwargs', {})
        self._node_names = None
        self._fh = None
        self._writer = None
        self.complib = complib
        self.complevel = complevel

    @classmethod
    def load(cls, model, data):
        url = data.pop("url")
        if not os.path.isabs(url) and model.path is not None:
            url = os.path.join(model.path, url)
        return cls(model, url, **data)

    def setup(self):
        """
        Setup the CSV file recorder.
        """

        if self.nodes is None:
            self._node_names = sorted(self.model.nodes.keys())
        else:
            node_names = []
            for node_ in self.nodes:
                # test if the node name is provided
                if isinstance(node_, str):
                    # lookup node by name
                    node_names.append(node_)
                else:
                    node_names.append((node_[1].name))
            self._node_names = node_names

    def reset(self):
        kwargs = {"newline": "", "encoding": "utf-8"}
        mode = "wt"

        if self.complib == "gzip":
            self._fh = gzip.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib in ("bz2", "bzip2"):
            self._fh = bz2.open(self.csvfile, mode, self.complevel, **kwargs)
        elif self.complib is None:
            self._fh = open(self.csvfile, mode, **kwargs)
        else:
            raise KeyError("Unexpected compression library: {}".format(self.complib))
        self._writer = csv.writer(self._fh, **self.csv_kwargs)
        # Write header data
        row = ["Datetime"] + [name for name in self._node_names]
        self._writer.writerow(row)

    def after(self):
        """
        Write the node values to the CSV file
        """
        values = [self.model.timestepper.current.datetime.isoformat()]
        for node_name in self._node_names:
            node = self.model.nodes[node_name]
            if isinstance(node, AbstractStorage):
                values.append(node.volume[self.scenario_index])
            elif isinstance(node, AbstractNode):
                
                values.append(node.flow[self.scenario_index-1])
            else:
                raise ValueError("Unrecognised Node type '{}' for CSV writer".format(type(node)))

        self._writer.writerow(values)

    def finish(self):
        if self._fh:
            self._fh.close()
CSVRecorder.register()

class HydropowerTargetParameterIndus(Parameter):
    """ A parameter that returns flow from a hydropower generation target.

    Same as HydropowerTargetParameter, except that the head unit is converted from feet to meters.

    """
    def __init__(self, model, target, water_elevation_parameter=None, max_flow=None, min_flow=None,
                 turbine_elevation=0.0, efficiency=1.0, density=1000, min_head=0.0,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerTargetParameterIndus, self).__init__(model, **kwargs)

        self.target = target
        self.water_elevation_parameter = water_elevation_parameter
        self.max_flow = max_flow
        self.min_flow = min_flow
        self.min_head = min_head
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion

    def value(self, ts, scenario_index):
        power = self.target.get_value(scenario_index)

        if self.water_elevation_parameter is not None:
            head = self.water_elevation_parameter.get_value(scenario_index)
            if self.turbine_elevation is not None:
                head -= self.turbine_elevation
        elif self.turbine_elevation is not None:
            head = self.turbine_elevation
        else:
            raise ValueError('One or both of storage_node or level must be set.')

        # -ve head is not valid
        head = max(head, 0.0)

        head = head * 0.3048 # feet to metre

        # Apply minimum head threshold.
        if head < self.min_head:
            return 0.0

        # Get the flow from the current node
        q = inverse_hydropower_calculation(power, head, 0.0, self.efficiency, density=self.density,
                                           flow_unit_conversion=self.flow_unit_conversion,
                                           energy_unit_conversion=self.energy_unit_conversion)

        # Bound the flow if required
        if self.max_flow is not None:
            q = min(self.max_flow.get_value(scenario_index), q)
        if self.min_flow is not None:
            q = max(self.min_flow.get_value(scenario_index), q)

        assert q >= 0.0

        return q

    @classmethod
    def load(cls, model, data):

        target = load_parameter(model, data.pop("target"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        if "max_flow" in data:
            max_flow = load_parameter(model, data.pop("max_flow"))
        else:
            max_flow = None

        if "min_flow" in data:
            min_flow = load_parameter(model, data.pop("min_flow"))
        else:
            min_flow = None

        return cls(model, target, water_elevation_parameter=water_elevation_parameter,
                   max_flow=max_flow, min_flow=min_flow, **data)
HydropowerTargetParameterIndus.register()

class HydropowerRecorderIndus(NumpyArrayNodeRecorder):
    """ Calculates the power production using the hydropower equation

    Same as HydropowerRecorder, except that the head unit is converted from feet to meters.

    """
    
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(HydropowerRecorderIndus, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        
    def setup(self):

        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self._data = np.zeros((nts, ncomb))
        
    def reset(self):

        self._data[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self.water_elevation_parameter is not None:
                head = self.water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)

            head = head * 0.3048 # feet to metre

            # Get the flow from the current node
            q = self.node.flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._data[ts.index, scenario_index.global_id] = power
            
        return 0
        
    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self._data, axis=0, ignore_nan=self.ignore_nan)
        
    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self._data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
HydropowerRecorderIndus.register()

class TotalHydroEnergyRecorderIndus(BaseConstantNodeRecorder):
    """ Calculates the total energy production using the hydropower equation from a model run.

    Same as TotalHydroEnergyRecorder, except that the head unit is converted from feet to meters.

    """
    def __init__(self, model, node, water_elevation_parameter=None, turbine_elevation=0.0, efficiency=1.0, density=1000,
                 flow_unit_conversion=1.0, energy_unit_conversion=1e-6, **kwargs):
        super(TotalHydroEnergyRecorderIndus, self).__init__(model, node, **kwargs)

        self.water_elevation_parameter = water_elevation_parameter
        self.turbine_elevation = turbine_elevation
        self.efficiency = efficiency
        self.density = density
        self.flow_unit_conversion = flow_unit_conversion
        self.energy_unit_conversion = energy_unit_conversion
        
    def setup(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))
        
    def reset(self):
        self._values[...] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        days = ts.days
        flow = self.node.flow

        for scenario_index in self.model.scenarios.combinations:

            if self.water_elevation_parameter is not None:
                head = self.water_elevation_parameter.get_value(scenario_index)
                if self.turbine_elevation is not None:
                    head -= self.turbine_elevation
            elif self.turbine_elevation is not None:
                head = self.turbine_elevation
            else:
                raise ValueError('One or both of storage_node or level must be set.')

            # -ve head is not valid
            head = max(head, 0.0)

            head = head * 0.3048 # feet to metre

            # Get the flow from the current node
            q = self.node.flow[scenario_index.global_id]
            power = hydropower_calculation(q, head, 0.0, self.efficiency, density=self.density,
                                             flow_unit_conversion=self.flow_unit_conversion,
                                             energy_unit_conversion=self.energy_unit_conversion)

            self._values[scenario_index.global_id] += power * days * 24
            
        return 0
            
    def values(self):
        return self._values

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        if "water_elevation_parameter" in data:
            water_elevation_parameter = load_parameter(model, data.pop("water_elevation_parameter"))
        else:
            water_elevation_parameter = None

        return cls(model, node, water_elevation_parameter=water_elevation_parameter, **data)
TotalHydroEnergyRecorderIndus.register()

class renewable_dataframe(Parameter):
 
    def __init__(self, model, url, key, **kwargs):
        super().__init__(model, **kwargs)
        self.profile = url
        self.key = key
       
    def setup(self):
        super().setup()
        self.profile = pd.read_hdf(self.profile, key=self.key)
        self.hour = 0
 
    def value(self, ts, scenario_index):
       
        ts = self.model.timestepper.current
        month = ts.month
        day = ts.day
       
        if self.hour == 24:
            self.hour = 0
           
        hour = self.hour
       
        index = str(month) + '-' + str(day) + '-' + str(hour)
 
        # Make sure index exists
        if index not in self.profile.index:
            raise KeyError(f"Index {index} not found in profile HDF5")
 
        #print(self.profile)
       
        profile = self.profile.loc[index]
       
        self.hour += 1
       
        return profile
 
    @classmethod
    def load(cls, model, data):
 
        url = data.pop("url")
        key = data.pop("key")
 
        return cls(model, url, key, **data)
renewable_dataframe.register()