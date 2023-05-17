import numpy as np
import pandas as pd

#from pywr.parameters import load_parameter
from pywr.recorders import NumpyArrayNodeRecorder, NodeRecorder, Aggregator, NumpyArrayStorageRecorder


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