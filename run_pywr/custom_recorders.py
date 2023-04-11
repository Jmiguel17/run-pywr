import numpy as np
import pandas as pd

from pywr.parameters import load_parameter
from pywr.recorders import MeanParameterRecorder, NumpyArrayNodeRecorder, NodeRecorder, Aggregator, BaseConstantParameterRecorder


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