"""
Custom Pywr recorders for model evaluation and water system performance metrics.

This module groups recorders into four families:

1) Comparison / calibration metrics against observed data
   - Root mean squared error (RMSE)
   - Nash-Sutcliffe efficiency (NSE)
   - Percent bias (PBIAS) and absolute PBIAS

   These follow common hydrologic model evaluation guidance (e.g., Moriasi et al., 2007).

2) Economic / value metrics
   - Annual irrigation revenue based on annual curtailment ratio
   - Annual hydropower energy or revenue based on Pywr's `hydropower_calculation`

3) Robustness / persistence (annual deficit run-length family)
   - Annual deficit-year frequency
   - Annual deficit episode count (runs >= K years)
   - Annual deficit episode excess years (persistence-weighted)
   - Annual maximum consecutive deficit years (worst-case persistence)
   - Storage-node equivalents using volume thresholds

4) Reliability / resilience (period-failure family)
   - Monthly and annual storage-threshold reliability
   - Monthly demand-satisfaction reliability
   - Annual demand-satisfaction reliability
   - Resilience as probability of recovery after failure (monthly/annual for demand nodes; monthly for storage nodes)

Design notes
------------
- Recorders are written to align with Pywr recorder conventions:
  * `setup()` allocates arrays, `after()` collects timestep data, and `finish()` computes derived metrics.
  * `values()` returns a vector with length equal to the number of scenarios.
  * `aggregated_value()` aggregates across scenarios using `agg_func` when available.

- Wherever feasible, the implementation uses Pywr's built-in `Aggregator` for temporal and scenario aggregation.

- Units are not implicitly converted unless parameters explicitly specify conversion factors. For any recorder that
  compares against observed data, observed and simulated values must be in consistent units (and this must be checked
  as part of model QA/QC).

"""
import numpy as np
import pandas as pd

from datetime import timedelta

from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NodeRecorder, Aggregator, hydropower_calculation
from pywr.parameters import load_parameter


class MetricRecorderMixin:
    """
    Mixin providing common functionality for "comparison recorders" that compute
    a scalar (or per-scenario) performance metric between:

      - an observed time series loaded from file; and
      - a simulated time series stored by a `NumpyArray*Recorder`.

    Design notes
    ------------
    - This mixin centralises time alignment and merging logic so metric-specific
      recorders only implement `calculate_metric(obs, sim)`.
    - The simulated data are taken from `self.data` (recorded during the run).
    - The observed data are provided at construction time via the `observed`
      keyword (loaded by `load`).
    - Alignment is prepared during `setup()` and performed during `finish()`,
      enabling `values()` to remain numeric and lightweight.

    Units requirement (must be checked)
    -----------------------------------
    Observed values MUST be in the same physical units as the simulated values
    stored by this recorder (e.g., both in m³/s, or both in Mm³ per month).
    This implementation enforces a runtime scale sanity-check by comparing the
    median absolute magnitude of the aligned observed and simulated series.

    If the magnitude ratio is outside `units_check_ratio_bounds`, a ValueError is
    raised, because a unit mismatch (or inconsistent aggregation such as mean vs sum)
    is the most common cause of invalid metrics.

    You can disable or adjust this check via:
        - units_check=False
        - units_check_ratio_bounds=(lower, upper)

    Parameters
    ----------
    observed : pandas.Series
        Observed values indexed by datetime-like values (or values convertible
        to datetime). The index should represent the comparison timestep scale
        unless you deliberately choose to only resample the model to match.
    obs_freq : str or None
        Optional pandas offset alias used to resample model outputs prior to
        comparison (e.g., "M", "MS", "D"). If None, no resampling is performed.
    units_check : bool
        If True, enforce a scale sanity-check after alignment is built in `finish()`.
    units_check_ratio_bounds : tuple[float, float]
        (lower, upper) bounds for acceptable ratio of observed/simulated magnitude.
        Default is (1e-3, 1e3). Ratios outside this range raise a ValueError.

    Expected shapes
    ---------------
    - Observed aligned array: (T,)
    - Simulated aligned array: (T, S) where S is the number of scenarios

    Missing values policy
    ---------------------
    Metrics are computed using pairwise deletion per scenario:
      - For scenario s, only timesteps with finite obs AND finite sim[:, s]
        contribute to that scenario’s metric.
      - If there are insufficient valid points, that scenario’s metric is NaN.

    Subclass contract
    -----------------
    Subclasses must implement:
        calculate_metric(self, obs: np.ndarray, sim: np.ndarray) -> np.ndarray|float
    """

    def __init__(
        self,
        *args,
        observed=None,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """
        Store observed time series and comparison configuration.

        Notes
        -----
        This mixin does not call `super().__init__()` because the concrete base
        recorders (`NumpyArrayNodeRecorder`, `NumpyArrayStorageRecorder`) are
        initialised explicitly in the base comparison classes.
        """
        self.observed_original = observed
        self.obs_freq = obs_freq

        self.units_check = bool(units_check)
        self.units_check_ratio_bounds = tuple(units_check_ratio_bounds)

        # Prepared in setup()
        self._obs_df_prepared = None
        self._model_native_index = None

        # Cached alignment (built in finish())
        self._cached_obs_aligned = None
        self._cached_sim_aligned = None
        self._alignment_cache_ready = False

        # Cached metric values (computed on first values() call after finish())
        self._cached_metric_values = None

    def reset(self):
        """
        Reset recorder state for a new run.

        This clears any cached alignment and metric values, and then delegates
        to the parent recorder's `reset()` method.
        """
        self._cached_obs_aligned = None
        self._cached_sim_aligned = None
        self._alignment_cache_ready = False
        self._cached_metric_values = None

        super().reset()

    def setup(self):
        """
        Recorder setup hook.

        Improvements implemented
        ------------------------
        - Observed preprocessing is performed here (index parsing/coercion),
          aligning with Pywr's lifecycle expectations.
        - Model datetime index is also prepared here.

        The final merge/alignment is built in `finish()` once model data are complete.
        """
        super().setup()

        # Prepare observed once (do not mutate original).
        obs_df = self.observed_original.copy().to_frame()

        # Ensure datetime-like index if possible.
        if not pd.api.types.is_datetime64_any_dtype(obs_df.index):
            obs_df.index = pd.to_datetime(obs_df.index)

        # Standardise dtype to numeric (coerce errors to NaN).
        obs_df.iloc[:, 0] = pd.to_numeric(obs_df.iloc[:, 0], errors="coerce")

        self._obs_df_prepared = obs_df

        # Prepare model native datetime index once.
        native_index = self.model.timestepper.datetime_index
        if isinstance(native_index, pd.PeriodIndex):
            native_index = native_index.to_timestamp()
        else:
            # Ensure datetime64 dtype where possible.
            if not pd.api.types.is_datetime64_any_dtype(native_index):
                native_index = pd.to_datetime(native_index)

        self._model_native_index = native_index

    def finish(self):
        """
        Recorder finish hook.

        Improvements implemented
        ------------------------
        - Alignment/merge is performed once per run (after `self.data` is populated).
        - Alignment results are cached for subsequent `values()` calls.
        - A unit-consistency (scale) check is enforced here.

        Notes
        -----
        This method is called after the model run completes, which is the earliest
        point at which `self.data` contains the full simulated time series.
        """
        super().finish()

        self._build_alignment_cache()
        self._enforce_units_check()

    def _build_alignment_cache(self):
        """
        Build and cache aligned observed and simulated arrays.

        Alignment strategy (as implemented)
        ----------------------------------
        1. Use the preprocessed observed DataFrame from `setup()`.
        2. Construct a model DataFrame from `self.data` using the model's
           native datetime index.
        3. If `obs_freq` is provided, resample the model DataFrame using mean.
           If the model timestep frequency differs from `obs_freq`, convert
           indices to "YYYY-MM" strings to force month-based alignment.
        4. Merge observed and model DataFrames on index (inner join).
        5. Drop rows where observed is not finite (never usable for metrics).

        Cached outputs
        --------------
        self._cached_obs_aligned : np.ndarray | None
            1D array of aligned observed values (T,).
        self._cached_sim_aligned : np.ndarray | None
            2D array of aligned simulated values (T, S).
        """
        freq = self.obs_freq

        obs = self._obs_df_prepared
        if obs is None:
            # setup() not called or failed
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
            self._alignment_cache_ready = True
            return

        # Build model DataFrame from recorded data.
        mod_df = pd.DataFrame(self.data, index=self._model_native_index)

        # Apply resampling / index coercion logic.
        if freq is None:
            # No resampling requested
            obs_keyed = obs
            mod_keyed = mod_df
        else:
            if self.model.timestepper.freq == freq:
                mod_df = mod_df.resample(freq).mean()
                obs_keyed = obs
                mod_keyed = mod_df
            else:
                # Resample model to requested frequency then key both sides by YYYY-MM.
                mod_df.index = mod_df.index.astype("datetime64[ns]")
                mod_df = mod_df.resample(freq).mean()
                mod_df.index = mod_df.index.strftime("%Y-%m")

                obs_keyed = obs.copy()
                obs_keyed.index = obs_keyed.index.astype("datetime64[ns]").strftime("%Y-%m")

                mod_keyed = mod_df

        merged = pd.merge(obs_keyed, mod_keyed, how="inner", left_index=True, right_index=True)

        if merged.empty:
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
            self._alignment_cache_ready = True
            return

        obs_aligned = merged.iloc[:, 0].to_numpy()
        sim_aligned = merged.iloc[:, 1:].to_numpy()

        # Improvement 3 (partial): remove rows where observed is NaN/inf.
        # (Scenario-specific pairwise deletion is handled in metric calculations.)
        mask_obs = np.isfinite(obs_aligned)
        obs_aligned = obs_aligned[mask_obs]
        sim_aligned = sim_aligned[mask_obs, :]

        if obs_aligned.size == 0:
            self._cached_obs_aligned = None
            self._cached_sim_aligned = None
        else:
            self._cached_obs_aligned = obs_aligned
            self._cached_sim_aligned = sim_aligned

        self._alignment_cache_ready = True

    def _enforce_units_check(self):
        """
        Enforce a basic unit/scale sanity check.

        Rationale
        ---------
        The metrics assume observed and simulated data are directly comparable.
        If the observed series is in different units (or different aggregation,
        e.g., monthly total vs monthly mean), metrics are invalid.

        Implementation
        --------------
        Compare the median absolute magnitude of observed and simulated aligned data.
        If ratio outside `units_check_ratio_bounds`, raise ValueError.

        Notes
        -----
        This is a sanity check, not a physical unit conversion mechanism.
        """
        if not self.units_check:
            return

        if not self._alignment_cache_ready:
            return

        obs = self._cached_obs_aligned
        sim = self._cached_sim_aligned

        if obs is None or sim is None:
            return

        lower, upper = self.units_check_ratio_bounds

        # Robust scale estimators
        scale_obs = np.nanmedian(np.abs(obs))

        # For simulated, take median across timesteps and scenarios then median across scenarios.
        scale_sim_per_scenario = np.nanmedian(np.abs(sim), axis=0)
        scale_sim = np.nanmedian(scale_sim_per_scenario)

        # If either series is (effectively) all zeros, scale check is not informative.
        if not np.isfinite(scale_obs) or not np.isfinite(scale_sim) or scale_obs == 0 or scale_sim == 0:
            return

        ratio = scale_obs / scale_sim

        if ratio < lower or ratio > upper:
            raise ValueError(
                "Observed and simulated series appear to be on different scales. "
                "This usually indicates a unit mismatch (e.g., m³/s vs Mm³/month) or "
                "inconsistent aggregation (e.g., monthly mean vs monthly sum). "
                f"Scale ratio (obs/sim)={ratio:.3g} is outside bounds {self.units_check_ratio_bounds}. "
                "Verify units and aggregation are consistent, or adjust/disable the check via "
                "`units_check=False` or `units_check_ratio_bounds=(lower, upper)`."
            )

    def _get_merged_data(self):
        """
        Return cached aligned observed and simulated arrays.

        Improvements implemented
        ------------------------
        - Alignment is built once in `finish()` and cached.
        - If cache is not ready (e.g., finish not called), build it on demand.

        Returns
        -------
        obs_aligned : np.ndarray or None
            1D array of aligned observed values (T,).
        sim_aligned : np.ndarray or None
            2D array of aligned simulated values (T, S).
        """
        if self._alignment_cache_ready:
            return self._cached_obs_aligned, self._cached_sim_aligned

        # On-demand build (defensive), then return.
        self._build_alignment_cache()
        return self._cached_obs_aligned, self._cached_sim_aligned

    def values(self):
        """
        Compute the metric for this recorder.

        Returns
        -------
        np.ndarray
            A 1D numpy array of metric values per scenario, shape (S,).
            If the metric calculation returns a scalar, it is wrapped as (1,).

        Improvements implemented
        ------------------------
        - Result caching: subsequent calls return cached metric values without
          recomputing alignment or metrics.
        - Numeric-only: alignment is expected to be precomputed in `finish()`.

        Notes
        -----
        - If alignment yields no overlapping timestamps, returns [np.nan].
        - Subclasses must implement `calculate_metric(obs, sim)`.
        """
        if self._cached_metric_values is not None:
            return self._cached_metric_values

        obs, sim = self._get_merged_data()

        if obs is None or sim is None:
            self._cached_metric_values = np.array([np.nan])
            return self._cached_metric_values

        val = self.calculate_metric(obs, sim)

        if np.isscalar(val):
            self._cached_metric_values = np.array([val])
        else:
            self._cached_metric_values = np.array(val)

        return self._cached_metric_values

    # def aggregated_value(self):
    #     """
    #     Return a single scalar metric value.

    #     Behaviour
    #     ---------
    #     - If there is a single scenario value, return it as a float.
    #     - If multiple scenarios are present, return the mean across scenarios.

    #     Returns
    #     -------
    #     float
    #         Scalar aggregated metric value.
    #     """
    #     val_array = self.values()

    #     if val_array.size == 1:
    #         return float(val_array[0])

    #     return float(np.mean(val_array))

    @classmethod
    def load(cls, model, data):
        """
        Pywr JSON loader for comparison recorders.

        Expected JSON fields (in `data`)
        --------------------------------
        observed : str
            Column name in the source file containing observed values.
        index_col : str
            Column name in the source file containing the datetime index.
        url : str
            Path/URL to the observed data file (.csv, .xlsx, .xls).
        obs_freq : str, optional
            Resampling frequency (pandas offset alias).
        node : str|dict
            Node reference resolvable by `model._get_node_from_ref`.

        Optional JSON fields (in `data`)
        --------------------------------
        units_check : bool
            Enable/disable the enforced unit/scale sanity check.
        units_check_ratio_bounds : list[float, float] or tuple[float, float]
            Acceptable bounds for observed/simulated magnitude ratio.

        Returns
        -------
        cls
            Instantiated recorder object.

        Important
        ---------
        Observed values MUST be in the same units as the simulated values recorded
        by this recorder. This implementation does not perform unit conversion.
        """
        observed_key = data.pop("observed")
        index_col = data.pop("index_col")
        url = data.pop("url")
        obs_freq = data.pop("obs_freq", None)

        # Optional enforcement configuration
        units_check = data.pop("units_check", True)
        units_check_ratio_bounds = data.pop("units_check_ratio_bounds", (1e-3, 1e3))

        if ".csv" in url:
            df_raw = pd.read_csv(url)
        elif ".xlsx" in url or ".xls" in url:
            df_raw = pd.read_excel(url)
        else:
            raise ValueError(f"Unsupported file format: {url}")

        if index_col not in df_raw.columns:
            raise KeyError(f"Index column '{index_col}' not found in file.")

        df_raw[index_col] = pd.to_datetime(df_raw[index_col])
        df_raw.set_index(index_col, inplace=True)

        observed_series = pd.to_numeric(df_raw[observed_key], errors="coerce")
        node = model._get_node_from_ref(model, data.pop("node"))

        # Ensure bounds is a tuple (JSON likely provides list)
        if isinstance(units_check_ratio_bounds, list):
            units_check_ratio_bounds = tuple(units_check_ratio_bounds)

        return cls(
            model,
            node,
            observed=observed_series,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
            **data,
        )


class BaseComparisonNodeRecorder(MetricRecorderMixin, NumpyArrayNodeRecorder):
    """
    Base comparison recorder for Node flows.

    Inherits storage of simulated node timeseries from `NumpyArrayNodeRecorder`
    and metric computation/alignment behaviour from `MetricRecorderMixin`.
    """

    def __init__(
        self,
        model,
        node,
        observed,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """Initialise with observed series and optional comparison configuration."""
        MetricRecorderMixin.__init__(
            self,
            observed=observed,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
        )
        NumpyArrayNodeRecorder.__init__(self, model, node, **kwargs)


class BaseComparisonStorageRecorder(MetricRecorderMixin, NumpyArrayStorageRecorder):
    """
    Base comparison recorder for Storage node volumes.

    Inherits storage of simulated storage timeseries from `NumpyArrayStorageRecorder`
    and metric computation/alignment behaviour from `MetricRecorderMixin`.
    """

    def __init__(
        self,
        model,
        node,
        observed,
        obs_freq=None,
        units_check=True,
        units_check_ratio_bounds=(1e-3, 1e3),
        **kwargs,
    ):
        """Initialise with observed series and optional comparison configuration."""
        MetricRecorderMixin.__init__(
            self,
            observed=observed,
            obs_freq=obs_freq,
            units_check=units_check,
            units_check_ratio_bounds=units_check_ratio_bounds,
        )
        NumpyArrayStorageRecorder.__init__(self, model, node, **kwargs)


class RootMeanSquaredErrorNodeRecorder(BaseComparisonNodeRecorder):
    """
    RMSE metric for Node flows.

    RMSE = sqrt( mean( (obs - sim)^2 ) )

    Returns per-scenario RMSE values.

    Missing values
    --------------
    Pairwise deletion is applied implicitly: any timestep with NaN in either
    obs or sim for scenario s is ignored for that scenario.
    """

    def calculate_metric(self, obs, sim):
        # obs: (T,), sim: (T, S)
        err2 = (obs[:, np.newaxis] - sim) ** 2
        return np.sqrt(np.nanmean(err2, axis=0))


RootMeanSquaredErrorNodeRecorder.register()


class RootMeanSquaredErrorStorageRecorder(BaseComparisonStorageRecorder):
    """
    RMSE metric for Storage node volumes.

    RMSE = sqrt( mean( (obs - sim)^2 ) )

    Returns per-scenario RMSE values.

    Missing values
    --------------
    Pairwise deletion is applied implicitly: any timestep with NaN in either
    obs or sim for scenario s is ignored for that scenario.
    """

    def calculate_metric(self, obs, sim):
        err2 = (obs[:, np.newaxis] - sim) ** 2
        return np.sqrt(np.nanmean(err2, axis=0))


RootMeanSquaredErrorStorageRecorder.register()


class NashSutcliffeEfficiencyNodeRecorder(BaseComparisonNodeRecorder):
    """
    Nash–Sutcliffe Efficiency (NSE) metric for Node flows.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Returns per-scenario NSE values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are used for numerator, mean(obs), and denominator.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 2:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            o_mean = np.mean(o)
            numerator = np.sum((o - m) ** 2)
            denominator = np.sum((o - o_mean) ** 2)

            out[s] = np.nan if denominator == 0 else (1.0 - (numerator / denominator))

        return out


NashSutcliffeEfficiencyNodeRecorder.register()


class NashSutcliffeEfficiencyStorageRecorder(BaseComparisonStorageRecorder):
    """
    Nash–Sutcliffe Efficiency (NSE) metric for Storage node volumes.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Returns per-scenario NSE values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are used for numerator, mean(obs), and denominator.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 2:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            o_mean = np.mean(o)
            numerator = np.sum((o - m) ** 2)
            denominator = np.sum((o - o_mean) ** 2)

            out[s] = np.nan if denominator == 0 else (1.0 - (numerator / denominator))

        return out


NashSutcliffeEfficiencyStorageRecorder.register()


class AbsolutePercentBiasNodeRecorder(BaseComparisonNodeRecorder):
    """
    Absolute Percent Bias (|PBIAS|) metric for Node flows.

    |PBIAS| = abs( sum(obs - sim) / sum(obs) ) * 100

    Returns per-scenario absolute percent bias values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are included in sums.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 1:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            sum_obs = np.sum(o)
            if sum_obs == 0:
                out[s] = np.nan
                continue

            out[s] = np.abs(np.sum(o - m) * 100.0 / sum_obs)

        return out


AbsolutePercentBiasNodeRecorder.register()


class AbsolutePercentBiasStorageRecorder(BaseComparisonStorageRecorder):
    """
    Absolute Percent Bias (|PBIAS|) metric for Storage node volumes.

    |PBIAS| = abs( sum(obs - sim) / sum(obs) ) * 100

    Returns per-scenario absolute percent bias values.

    Missing values
    --------------
    Pairwise deletion per scenario:
      - For each scenario s, only timesteps where obs and sim[:, s] are finite
        are included in sums.
    """

    def calculate_metric(self, obs, sim):
        n_scen = sim.shape[1]
        out = np.full(n_scen, np.nan, dtype=float)

        for s in range(n_scen):
            mask = np.isfinite(obs) & np.isfinite(sim[:, s])
            if np.count_nonzero(mask) < 1:
                out[s] = np.nan
                continue

            o = obs[mask]
            m = sim[mask, s]

            sum_obs = np.sum(o)
            if sum_obs == 0:
                out[s] = np.nan
                continue

            out[s] = np.abs(np.sum(o - m) * 100.0 / sum_obs)

        return out


AbsolutePercentBiasStorageRecorder.register()


class AnnualIrrigationRevenueRecorder(NodeRecorder):
    """
    Annual irrigation revenue per scenario derived from annual supply/demand curtailment.

    This new recorder replace the old "AverageAnnualIrrigationRevenueScenarioRecorder" which is in the run-wb-models branch.

    Purpose
    -------
    This recorder produces (1) an annual time series of irrigation revenue for each scenario,
    and (2) a single scalar value per scenario by aggregating the annual revenues over time.

    The recorder is designed for irrigation nodes where:
      - the node supply is represented by `node.flow` (per scenario), and
      - the irrigation demand is represented by the node's `max_flow` parameter (or a
        supplied `demand_parameter` override).

    The motivation for annual aggregation is to make objective functions stable across
    different model timestep granularities (e.g., monthly vs daily), provided that the
    underlying units are consistent and timestep totals are computed correctly.

    Core calculation
    ----------------
    Revenue is computed from a curtailment ratio derived on annual totals:

        annual_supply  = sum_t( supply_rate_t  * timestep_days_t )     if flow_is_per_day
        annual_demand  = sum_t( demand_rate_t  * timestep_days_t )     if flow_is_per_day
        r_y            = annual_supply / annual_demand

    Annual crop yield (kg) and annual revenue (M$) are then computed as:

        crop_yield_kg  = r_y * area_ha * yield_per_area_kg_ha
        revenue_M$     = (crop_yield_kg / 1000) * price_$/t / 1e6

    Key assumptions and required unit consistency
    ---------------------------------------------
    The recorder does NOT perform unit conversion. You must ensure that all relevant
    quantities are physically consistent:

    1) Supply and demand rates:
       - `node.flow` is assumed to be a per-day rate (flow_is_per_day=True),
         meaning it must be multiplied by `timestep.days` to obtain a timestep total.
       - `demand_parameter` (or `node.max_flow`) must be on the same per-day basis if
         flow_is_per_day=True. This matches the common pattern where irrigation demand
         is returned as Mm3/day by a parameter.

       If your model uses per-timestep totals already, set flow_is_per_day=False to prevent
       multiplying by `timestep.days`.

    2) Yield and area:
       - `area` must be in hectares (ha).
       - `yield_per_area` must be in kg/ha.

    3) Price:
       - `price` must be in dollars per metric tonne ($/t).

    4) Output units:
       - Revenue is returned in million dollars (M$).

    Curtailment ratio semantics
    ---------------------------
    Curtailment ratio r_y is computed on ANNUAL totals:

        r_y = annual_supply / annual_demand

    Special cases:
    - annual_demand == 0:
        r_y is set to `zero_demand_ratio` (default 1.0).
        Rationale: for irrigation-demand parameters (e.g., FAO-based irrigation water requirement),
        demand may legitimately be zero (e.g., rainfall meets crop water needs). In that case, zero
        irrigation requirement should not force crop yield or revenue to zero.
    - clip_ratio:
        If True, r_y is clipped to [0, 1] to prevent over-supply inflating yield beyond the
        nominal maximum.

    Pywr integration and outputs
    ----------------------------
    - The recorder maintains annual accumulators during the run (in `after()`), then computes
      final annual revenue in `finish()`.
    - `to_dataframe()` returns the full annual revenue time series as a DataFrame with:
        index   = annual timestamps (YYYY-12-31),
        columns = model.scenarios.multiindex,
        values  = annual revenue in M$.
    - `values()` returns a 1D numpy array of length n_scenarios: scalar revenue value per scenario,
      computed by applying a temporal aggregation (`temporal_agg_func`) across the annual time axis.
    - Scenario aggregation across scenarios (if required by the optimisation framework) should use
      Pywr's standard `agg_func` mechanism (e.g., mean/median/min/max) applied by the base recorder
      machinery (Recorder.aggregated_value()).

    Configuration via JSON
    ----------------------
    This recorder can be instantiated from a Pywr JSON model file. The typical configuration is:

    Example 1: Minimal configuration (use node.max_flow for demand, area/yield from demand parameter)
    -----------------------------------------------------------------------------------------------
    {
      "my_revenue_recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "price": 2393,
        "temporal_agg_func": "mean",
        "agg_func": "mean"
      }
    }

    Example 2: Override area (Parameter reference) and use robust defaults
    ----------------------------------------------------------------------
    {
      "__AGR_TJK_Vakhsh_Cotton__:Average annual crop yield revenue recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "agg_func": "median",
        "temporal_agg_func": "median",
        "area": "__AGR_TJK_Vakhsh_Cotton__:area",
        "price": 2393,
        "zero_demand_ratio": 1.0,
        "clip_ratio": true,
        "flow_is_per_day": true
      }
    }

    Example 3: Override demand_parameter explicitly (optional)
    ----------------------------------------------------------
    {
      "my_revenue_recorder": {
        "type": "AnnualIrrigationRevenueRecorder",
        "node": "AGR_TJK_Vakhsh_Cotton",
        "demand_parameter": "__AGR_TJK_Vakhsh_Cotton__:max_flow",
        "price": 2393
      }
    }

    JSON fields
    -----------
    Required:
      - node: node reference (string or dict compatible with model._get_node_from_ref)

    Optional:
      - price: float or Parameter reference
      - temporal_agg_func: temporal aggregator name (e.g., "mean", "median", "min", "max")
      - agg_func: scenario aggregator name used by Pywr when requesting a single aggregated value
      - flow_is_per_day: bool; if True multiply supply/demand rates by timestep.days
      - zero_demand_ratio: float; ratio used when annual_demand == 0 (default 1.0)
      - clip_ratio: bool; clip ratio to [0, 1] (default True)
      - area: float or Parameter reference; if absent uses demand_parameter.area when available
      - yield_per_area: float or Parameter reference; if absent uses demand_parameter.yield_per_area when available
      - demand_parameter: float/Parameter reference (typically a Parameter); if absent uses node.max_flow

    Lifecycle and data flow (implementation overview)
    -------------------------------------------------
    - setup():
        * Builds a mapping from each model timestep to an annual "year slot".
        * Allocates annual accumulators for supply and demand with shape (n_years, n_scenarios).
        * Resolves the demand parameter source:
            - uses `demand_parameter` if provided, otherwise uses `node.max_flow`.
        * Resolves the sources for area and yield_per_area:
            - uses explicit overrides if provided, otherwise attempts to read
              `.area` and `.yield_per_area` from the demand parameter.
        * Clears cached scenario arrays and outputs for safety.

    - after():
        * Retrieves the current timestep and its annual slot (year index).
        * Lazily evaluates scenario-constant arrays (area, yield, price) the first time `after()` runs.
          This avoids evaluating parameters too early in the model lifecycle.
        * Retrieves per-scenario supply from `node.flow`.
        * Retrieves per-scenario demand from the demand parameter (vectorised if possible; otherwise per scenario).
        * Converts rates to timestep totals using timestep.days if flow_is_per_day=True.
        * Accumulates timestep totals into annual supply/demand arrays.

    - finish():
        * Computes annual curtailment ratios from annual totals.
        * Applies the zero-demand handling rule and optional clipping.
        * Computes annual crop yield (kg) and annual revenue (M$).
        * Caches both the annual revenue numpy array and a DataFrame view.

    - to_dataframe():
        * Returns the cached annual DataFrame (computes via finish() if needed).

    - values():
        * Returns per-scenario scalar values by temporally aggregating annual revenue (axis=0).

    Notes on robustness
    -------------------
    - The recorder avoids pandas resampling and instead uses an explicit year mapping derived from the
      model's timestep index. This reduces assumptions about timestep regularity (daily/monthly/etc.).
    - The `zero_demand_ratio` behaviour is critical for irrigation-demand parameters where irrigation
      demand can legitimately be zero without implying crop failure.

    """

    def __init__(
        self,
        model,
        node,
        price=1.0,
        temporal_agg_func="mean",
        flow_is_per_day=True,
        zero_demand_ratio=1.0,
        clip_ratio=True,
        area=None,
        yield_per_area=None,
        demand_parameter=None,
        **kwargs,
    ):
        """
        Initialise the recorder.

        Parameters
        ----------
        model : pywr.core.Model
            Pywr model instance.
        node : pywr.core.Node
            Node whose supply is evaluated using `node.flow`.
        price : float or Parameter, default 1.0
            Crop price in $/t. May be scenario-dependent if provided as a Parameter.
        temporal_agg_func : str, default "mean"
            Aggregation function applied over the annual time axis in `values()`
            (e.g., "mean", "median", "min", "max").
        flow_is_per_day : bool, default True
            If True, treat supply and demand as per-day rates and multiply by timestep.days
            to obtain timestep totals before annual aggregation.
        zero_demand_ratio : float, default 1.0
            Curtailment ratio used when annual demand is zero.
        clip_ratio : bool, default True
            If True, clip curtailment ratio to [0, 1].
        area : float or Parameter, optional
            Override for crop area (ha). If None, uses demand_parameter.area when available.
        yield_per_area : float or Parameter, optional
            Override for yield (kg/ha). If None, uses demand_parameter.yield_per_area when available.
        demand_parameter : Parameter, optional
            Override for irrigation demand source. If None, uses node.max_flow.

        Other Parameters
        ----------------
        **kwargs are passed to the NodeRecorder base class. This includes standard recorder
        options such as:
          - name
          - comment
          - ignore_nan
          - agg_func (scenario aggregation)
        """
        super().__init__(model, node, **kwargs)

        self.flow_is_per_day = bool(flow_is_per_day)
        self.zero_demand_ratio = float(zero_demand_ratio)
        self.clip_ratio = bool(clip_ratio)

        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        # Optional overrides (float or Parameter). If None, pulled from node.max_flow.
        self._area_src = area
        self._yield_src = yield_per_area
        self._price_src = price
        self._demand_src = demand_parameter  # defaults to node.max_flow if None

        # Cached scenario arrays (n_scen,), evaluated lazily on first `after()`
        self._area = None
        self._yield = None
        self._price = None

        # Annual accumulators (n_years, n_scen)
        self._year_labels = None
        self._year_index = None
        self._annual_supply = None
        self._annual_demand = None

        # Cached final annual revenue
        self._annual_revenue = None
        self._annual_df = None

        # Demand parameter resolved in setup
        self._demand_param = None
        self._resolved_area_src = None
        self._resolved_yield_src = None

    @staticmethod
    def _eval_scenario_array(model, src, n_scen):
        """
        Evaluate a float/Parameter into a (n_scen,) array.

        Why evaluation is delayed
        -------------------------
        Parameters in Pywr may depend on internal runtime state that is fully initialised
        only once the model is actively stepping through timesteps. Evaluating scenario arrays
        lazily (during/after the first `after()` call) avoids accessing parameter internals too early.

        Behaviour
        ---------
        - If src is a numeric scalar, returns a constant vector of length n_scen.
        - If src exposes get_all_values(), attempts vectorised evaluation.
        - Otherwise, falls back to calling get_value(scenario) for each scenario.

        Parameters
        ----------
        model : pywr.core.Model
            Pywr model.
        src : float or pywr.parameters.Parameter
            Value source.
        n_scen : int
            Number of scenario combinations.

        Returns
        -------
        numpy.ndarray
            A 1D array (n_scen,) containing the value for each scenario.
        """
        if isinstance(src, (int, float, np.floating)):
            return np.full(n_scen, float(src), dtype=float)

        # Prefer vectorised evaluation; if it fails, fall back to per-scenario loop.
        if hasattr(src, "get_all_values"):
            try:
                vals = np.asarray(src.get_all_values(), dtype=float)
                if vals.shape[0] == n_scen:
                    return vals
            except Exception:
                pass

        out = np.zeros(n_scen, dtype=float)
        for si in model.scenarios.combinations:
            out[si.global_id] = float(src.get_value(si))
        return out

    def setup(self):
        """
        Allocate annual accumulators and resolve parameter sources.

        This method:
          1) Builds an array mapping each timestep index to its year slot.
          2) Allocates annual supply/demand accumulators: shape (n_years, n_scenarios).
          3) Resolves the demand parameter:
               - uses explicit demand_parameter override if provided,
               - otherwise uses node.max_flow.
          4) Resolves the sources for area and yield_per_area (but does not evaluate them yet).
          5) Clears cached arrays and output caches.

        Raises
        ------
        AttributeError
            If demand cannot be resolved (no max_flow and no demand_parameter provided),
            or if area/yield cannot be resolved from overrides or the demand parameter.
        """
        super().setup()

        dt_index = self.model.timestepper.datetime_index
        if isinstance(dt_index, pd.PeriodIndex):
            dt_index = dt_index.to_timestamp()

        years = np.asarray([d.year for d in dt_index], dtype=int)
        self._year_labels = np.unique(years)
        year_to_slot = {y: i for i, y in enumerate(self._year_labels)}
        self._year_index = np.asarray([year_to_slot[y] for y in years], dtype=int)

        n_years = len(self._year_labels)
        n_scen = len(self.model.scenarios.combinations)

        self._annual_supply = np.zeros((n_years, n_scen), dtype=float)
        self._annual_demand = np.zeros((n_years, n_scen), dtype=float)

        # Resolve demand parameter (defaults to node.max_flow)
        if self._demand_src is not None:
            self._demand_param = self._demand_src
        else:
            mf = getattr(self.node, "max_flow", None)
            if mf is None:
                raise AttributeError("Node has no max_flow; provide demand_parameter in recorder config.")
            self._demand_param = mf

        # Resolve area/yield sources (but do NOT evaluate to arrays yet)
        self._resolved_area_src = self._area_src if self._area_src is not None else getattr(self._demand_param, "area", None)
        self._resolved_yield_src = self._yield_src if self._yield_src is not None else getattr(self._demand_param, "yield_per_area", None)

        if self._resolved_area_src is None or self._resolved_yield_src is None:
            raise AttributeError("Could not resolve area and/or yield_per_area for revenue calculation.")

        # Reset cached arrays/results
        self._area = None
        self._yield = None
        self._price = None
        self._annual_revenue = None
        self._annual_df = None

    def reset(self):
        """
        Reset annual accumulators and cached outputs for a new run.

        Notes
        -----
        This resets the annual supply/demand totals and clears cached outputs.
        It also clears cached scenario arrays (area, yield, price) to ensure correct
        behaviour if the model is reused for multiple runs.
        """
        super().reset()
        self._annual_supply[:, :] = 0.0
        self._annual_demand[:, :] = 0.0
        self._annual_revenue = None
        self._annual_df = None

        # Keep _area/_yield/_price cached across reset? Safer to clear for multi-run usage.
        self._area = None
        self._yield = None
        self._price = None

    def after(self):
        """
        Accumulate timestep supply/demand totals into annual totals for each scenario.

        Steps
        -----
        1) Determine the year slot for the current timestep.
        2) Lazily evaluate scenario-constant arrays for area, yield, and price on the first call.
        3) Retrieve per-scenario supply rate from `node.flow`.
        4) Retrieve per-scenario demand rate from the resolved demand parameter:
             - use get_all_values() when available; otherwise evaluate per scenario.
        5) Convert rates to timestep totals via timestep.days if flow_is_per_day=True.
        6) Accumulate into annual totals.

        Returns
        -------
        None
            Pywr recorder hooks typically return None; the original code returns nothing explicitly.
        """
        ts = self.model.timestepper.current
        y_i = self._year_index[ts.index]

        n_scen = self._annual_supply.shape[1]

        # Lazily evaluate scenario-constant arrays the first time a timestep exists
        if self._area is None:
            self._area = self._eval_scenario_array(self.model, self._resolved_area_src, n_scen)
        if self._yield is None:
            self._yield = self._eval_scenario_array(self.model, self._resolved_yield_src, n_scen)
        if self._price is None:
            self._price = self._eval_scenario_array(self.model, self._price_src, n_scen)

        w = float(ts.days) if self.flow_is_per_day else 1.0

        # Supply: node.flow is per scenario
        supply_rate = np.asarray(self.node.flow, dtype=float)
        if supply_rate.shape == ():  # scalar safeguard
            supply_rate = np.full(n_scen, float(supply_rate), dtype=float)

        # Demand: prefer get_all_values if available, otherwise loop
        if hasattr(self._demand_param, "get_all_values"):
            try:
                demand_rate = np.asarray(self._demand_param.get_all_values(), dtype=float)
            except Exception:
                demand_rate = np.zeros(n_scen, dtype=float)
                for si in self.model.scenarios.combinations:
                    demand_rate[si.global_id] = float(self._demand_param.get_value(si))
        else:
            demand_rate = np.zeros(n_scen, dtype=float)
            for si in self.model.scenarios.combinations:
                demand_rate[si.global_id] = float(self._demand_param.get_value(si))

        # Convert per-day rates to timestep totals
        self._annual_supply[y_i, :] += supply_rate * w
        self._annual_demand[y_i, :] += demand_rate * w

    def finish(self):
        """
        Compute annual revenue arrays and cache the outputs.

        This method:
          1) Validates that scenario-constant inputs were evaluated.
          2) Computes annual curtailment ratio using annual totals.
          3) Applies the zero-demand rule and handles NaN/inf safely.
          4) Optionally clips ratio to [0, 1].
          5) Computes annual crop yield (kg) and annual revenue (M$).
          6) Builds a DataFrame view indexed by year end.

        Raises
        ------
        RuntimeError
            If area/yield/price were not evaluated. This indicates `after()` did not run,
            typically because the model did not execute timesteps.
        """
        super().finish()

        if self._area is None or self._yield is None or self._price is None:
            raise RuntimeError(
                "Scenario-constant inputs (area/yield/price) were not evaluated. "
                "This typically means `after()` was never called (e.g., model did not run)."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                self._annual_demand == 0.0,
                self.zero_demand_ratio,
                self._annual_supply / self._annual_demand,
            )
            ratio = np.nan_to_num(ratio, nan=self.zero_demand_ratio, posinf=self.zero_demand_ratio, neginf=0.0)

        if self.clip_ratio:
            ratio = np.clip(ratio, 0.0, 1.0)

        # crop_yield_kg (years x scenarios), then revenue in M$
        crop_yield_kg = ratio * self._area[None, :] * self._yield[None, :]
        revenue = (crop_yield_kg / 1e3) * self._price[None, :] / 1e6

        self._annual_revenue = revenue

        sc_index = self.model.scenarios.multiindex
        annual_dt_index = pd.to_datetime([f"{y}-12-31" for y in self._year_labels])
        self._annual_df = pd.DataFrame(revenue, index=annual_dt_index, columns=sc_index)

    def to_dataframe(self):
        """
        Return annual irrigation revenue time series (M$) for each scenario.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by year-end timestamps with scenario MultiIndex columns.
            Values are annual revenues in M$.
        """
        if self._annual_df is None:
            self.finish()
        return self._annual_df

    def values(self):
        """
        Return per-scenario scalar by aggregating annual revenue over years.

        The aggregation is performed using Pywr's Aggregator on the annual revenue
        array produced in `finish()`.

        Returns
        -------
        numpy.ndarray
            1D array with length equal to the number of scenario combinations.
        """
        if self._annual_revenue is None:
            self.finish()

        return self._temporal_aggregator.aggregate_2d(
            self._annual_revenue,
            axis=0,  # aggregate over years
            ignore_nan=self.ignore_nan,
        )

    @classmethod
    def load(cls, model, data):
        """
        Create an AnnualIrrigationRevenueRecorder from a Pywr JSON definition.

        Parameters
        ----------
        model : pywr.core.Model
            Model instance.
        data : dict
            Parsed JSON dictionary for this recorder.

        Supported JSON keys
        -------------------
        - node (required): node reference for which supply is read from node.flow
        - price: float or Parameter reference (default 1.0)
        - temporal_agg_func: str (default "mean")
        - flow_is_per_day: bool (default True)
        - zero_demand_ratio: float (default 1.0)
        - clip_ratio: bool (default True)
        - area: float or Parameter reference (optional override)
        - yield_per_area: float or Parameter reference (optional override)
        - demand_parameter: Parameter reference (optional override; default uses node.max_flow)

        Notes
        -----
        Parameter references are resolved using `pywr.parameters.load_parameter`.
        """
        node = model._get_node_from_ref(model, data.pop("node"))

        def load_float_or_param(key, default=None):
            if key not in data:
                return default
            v = data.pop(key)
            if isinstance(v, (int, float)):
                return float(v)
            return load_parameter(model, v)

        return cls(
            model,
            node,
            price=load_float_or_param("price", 1.0),
            temporal_agg_func=data.pop("temporal_agg_func", "mean"),
            flow_is_per_day=data.pop("flow_is_per_day", True),
            zero_demand_ratio=data.pop("zero_demand_ratio", 1.0),
            clip_ratio=data.pop("clip_ratio", True),
            area=load_float_or_param("area", None),
            yield_per_area=load_float_or_param("yield_per_area", None),
            demand_parameter=load_float_or_param("demand_parameter", None),
            **data,
        )


AnnualIrrigationRevenueRecorder.register()


class AnnualHydroPowerRecorder(NodeRecorder):
    """
    Annual hydropower (energy) or hydropower-derived revenue recorder with optional seasonal month selection.

    Overview
    --------
    This recorder computes annual totals per scenario using Pywr's hydropower calculation. It is designed
    to be robust for non-daily timesteps (e.g., "7D") and avoids the common pattern of:
        timestep values -> daily resample/ffill -> month filter -> annual resample
    which can introduce allocation errors when timesteps straddle month boundaries.

    Instead, this recorder:
      1) Computes an energy-rate-per-day for each timestep using `hydropower_calculation`.
      2) Converts to energy over the timestep by multiplying by the number of days in the timestep.
      3) If a month-season filter is defined, splits each timestep at month boundaries and allocates energy
         by exact day fractions.
      4) Aggregates into annual totals per scenario.

    Crucially: removing the "zero year" completely
    ----------------------------------------------
    If you define a timestepper like:
        start = 2036-01-01
        end   = 2064-12-31
        timestep = 7D
    it is easy for custom annual recorders to accidentally include an extra year row (e.g., 2065) if they
    infer the year range from the last timestep end, rather than the configured end.

    This implementation uses the model timestepper definition directly:
      - Annual years are capped to [start.year, end.year] inclusive.
      - The last timestep is clipped so no contribution is counted after the configured model end date.

    That ensures you never get an "extra" year of all zeros from year-range inference.

    Head definition
    ---------------
    The head used in hydropower is computed as:
        head = water_elevation - turbine_elevation
    when `water_elevation_parameter` is provided. Negative head is clipped to zero.
    If no water elevation parameter is provided, `turbine_elevation` is treated as the head directly.

    Energy and revenue scaling (using `factor` as in Pywr)
    ------------------------------------------------------
    Many Pywr workflows apply a multiplicative scaling to recorder outputs. In some Pywr builds, `factor`
    is not accepted by `NodeRecorder.__init__`. Therefore, this recorder:
      - Pops `factor` from kwargs before calling `super().__init__`
      - Applies `factor` once, at the end, to annual totals.

    This supports the common pattern:
        revenue = energy * price
    where `factor` represents a price or conversion (e.g., M$ per MWh).

    Seasonal month selection
    ------------------------
    If `monthly_seasonality` is provided (list of months 1..12), only the portion of each timestep that
    lies within those months is included. This is done by splitting each timestep at calendar month
    boundaries and allocating energy by exact day fractions.

    JSON usage (general example)
    ----------------------------
    The recorder can be used directly from a Pywr JSON model. The following is a general example
    (names are placeholders):

    {
      "__TURBINE_NODE__:Annual Revenue": {
        "type": "AnnualHydroPowerRecorder",
        "node": "TURBINE_NODE_NAME",
        "agg_func": "mean",
        "temporal_agg_func": "mean",
        "efficiency": 0.95,
        "turbine_elevation": 100.0,
        "water_elevation_parameter": "RESERVOIR_LEVEL_PARAMETER",
        "flow_unit_conversion": 11.57407407,
        "energy_unit_conversion": 2.4e-05,
        "factor": 4.2e-05,
        "monthly_seasonality": [4, 5, 6, 7, 8, 9]
      }
    }

    Notes on conversions:
      - This recorder does not enforce a single unit system; it assumes your chosen conversion constants
        are consistent with your model's flow units and desired energy units.
      - With flow in million m³/day (Mm³/day), the pair:
            flow_unit_conversion   = 11.57407407  (= 1e6 / 86400)   -> Mm³/day to m³/s
            energy_unit_conversion = 2.4e-05      (= 86400 / 3.6e9) -> W to MWh/day
        is internally consistent for an output in MWh/day, which is then multiplied by days to give MWh.

    Outputs
    -------
    - to_dataframe(): annual totals (years x scenarios)
    - values(): per-scenario scalar from aggregating annual totals across years via `temporal_agg_func`

    """

    def __init__(
        self,
        model,
        node,
        monthly_seasonality=None,
        water_elevation_parameter=None,
        turbine_elevation=0.0,
        efficiency=1.0,
        density=1000.0,
        flow_unit_conversion=1.0,
        energy_unit_conversion=1e-6,
        **kwargs,
    ):
        # Pywr build compatibility: `factor` may not be accepted by NodeRecorder/Component __init__.
        # We consume it here and apply it later in `finish()`.
        self.factor = kwargs.pop("factor", None)

        # Temporal aggregation (over years) for values()
        temporal_agg_func = kwargs.pop("temporal_agg_func", "mean")

        # Initialise base recorder (handles name/comment/agg_func/ignore_nan/etc.)
        super().__init__(model, node, **kwargs)

        # Store temporal aggregator for values()
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        # Optional month filtering
        self._monthly_seasonality = monthly_seasonality

        # Optional elevation parameter used to compute head
        self._water_elevation_parameter = None
        self.water_elevation_parameter = water_elevation_parameter

        # Hydropower inputs
        self.turbine_elevation = float(turbine_elevation)
        self.efficiency = float(efficiency)
        self.density = float(density)
        self.flow_unit_conversion = float(flow_unit_conversion)
        self.energy_unit_conversion = float(energy_unit_conversion)

        # Model period (derived from timestepper in setup())
        self._model_start = None            # pd.Timestamp
        self._model_end = None              # pd.Timestamp (date)
        self._model_stop = None             # pd.Timestamp = model_end + 1 day (exclusive upper bound)

        # Timeline (timestep start timestamps)
        self._dt_starts = None              # list[pd.Timestamp]

        # Annual indexing
        self._year_labels = None            # np.ndarray of years
        self._year_to_slot = None           # dict year -> row index

        # Accumulated annual totals, and cached DataFrame view
        self._annual_totals = None          # shape (n_years, n_scen)
        self._annual_df = None              # DataFrame after finish()

        # Cache whether hydropower_calculation supports vectorised inputs in this environment
        self._hydro_vectorized = None

    @property
    def water_elevation_parameter(self):
        """Optional Parameter providing upstream water elevation used to compute head."""
        return self._water_elevation_parameter

    @water_elevation_parameter.setter
    def water_elevation_parameter(self, parameter):
        """
        Set water elevation parameter and register it as a dependency (child) so Pywr evaluates it correctly.
        """
        current = getattr(self, "_water_elevation_parameter", None)
        if current is not None and current in self.children:
            self.children.remove(current)
        if parameter is not None:
            self.children.add(parameter)
        self._water_elevation_parameter = parameter

    def setup(self):
        """
        Allocate arrays and build year indexing.

        The year range is capped using the model's timestepper definition:
            years = start.year .. end.year (inclusive)
        This prevents creation of an "extra" year row beyond the configured model end.
        """
        super().setup()

        # Read the model period from timestepper (preferred), otherwise fall back to the datetime_index.
        ts_start = getattr(self.model.timestepper, "start", None)
        ts_end = getattr(self.model.timestepper, "end", None)

        dt_index = self.model.timestepper.datetime_index
        if isinstance(dt_index, pd.PeriodIndex):
            dt_index = dt_index.to_timestamp()

        self._dt_starts = [pd.Timestamp(d) for d in dt_index]

        if ts_start is not None and ts_end is not None:
            self._model_start = pd.Timestamp(ts_start)
            self._model_end = pd.Timestamp(ts_end)
        else:
            # Fallback: infer from datetime_index; this is less precise than using explicit start/end.
            self._model_start = self._dt_starts[0]
            self._model_end = self._dt_starts[-1]

        # Use an exclusive upper bound for clipping: [start, end+1day)
        # This treats the JSON "end" as an inclusive date.
        self._model_stop = self._model_end + pd.Timedelta(days=1)

        # Build year labels strictly from the configured model period
        start_year = int(self._model_start.year)
        end_year = int(self._model_end.year)

        self._year_labels = np.arange(start_year, end_year + 1, dtype=int)
        self._year_to_slot = {y: i for i, y in enumerate(self._year_labels)}

        # Allocate annual totals array
        n_years = len(self._year_labels)
        n_scen = len(self.model.scenarios.combinations)
        self._annual_totals = np.zeros((n_years, n_scen), dtype=float)

        # Reset caches
        self._annual_df = None
        self._hydro_vectorized = None

    def reset(self):
        """Reset annual totals for a new model run."""
        super().reset()
        self._annual_totals[:, :] = 0.0
        self._annual_df = None

    def _iter_month_segments(self, dt_start, dt_end):
        """
        Yield (segment_start, segment_end) pairs that do not cross calendar month boundaries.

        This is the mechanism that makes seasonal slicing correct for 7D (or other) timesteps.
        """
        cur = dt_start
        while cur < dt_end:
            # First moment of next month
            next_month = (cur.to_period("M") + 1).to_timestamp()
            seg_end = next_month if next_month < dt_end else dt_end
            yield cur, seg_end
            cur = seg_end

    @staticmethod
    def _segment_days(seg_start, seg_end):
        """Return segment length in days (float)."""
        return (seg_end - seg_start).total_seconds() / 86400.0

    def _water_elevation_values(self, n_scen):
        """
        Retrieve water elevation per scenario for the current timestep.

        Uses vectorised evaluation if supported by the parameter in this environment.
        Falls back to per-scenario evaluation.
        """
        p = self._water_elevation_parameter
        if p is None:
            return None

        # Try vectorised access
        if hasattr(p, "get_all_values"):
            try:
                vals = np.asarray(p.get_all_values(), dtype=float)
                if vals.shape[0] == n_scen:
                    return vals
            except Exception:
                # Fallback to per-scenario evaluation
                pass

        out = np.zeros(n_scen, dtype=float)
        for si in self.model.scenarios.combinations:
            out[si.global_id] = float(p.get_value(si))
        return out

    def _hydropower_rate(self, q, head):
        """
        Compute hydropower output using Pywr's `hydropower_calculation`.

        Attempts vectorised evaluation for performance; falls back to per-scenario if required.
        The returned values are treated as an energy rate per day in units implied by
        `energy_unit_conversion`.
        """
        if self._hydro_vectorized is False:
            out = np.zeros_like(q, dtype=float)
            for i in range(q.shape[0]):
                out[i] = hydropower_calculation(
                    float(q[i]),
                    float(head[i]),
                    0.0,
                    self.efficiency,
                    density=self.density,
                    flow_unit_conversion=self.flow_unit_conversion,
                    energy_unit_conversion=self.energy_unit_conversion,
                )
            return out

        try:
            out = hydropower_calculation(
                q,
                head,
                0.0,
                self.efficiency,
                density=self.density,
                flow_unit_conversion=self.flow_unit_conversion,
                energy_unit_conversion=self.energy_unit_conversion,
            )
            self._hydro_vectorized = True
            return np.asarray(out, dtype=float)
        except Exception:
            self._hydro_vectorized = False
            return self._hydropower_rate(q, head)

    def after(self):
        """
        Accumulate annual totals for the current timestep.

        Steps:
          1) Determine timestep [dt_start, dt_end_raw)
          2) Clip dt_end to the configured model stop date (end+1day) to avoid counting beyond end.
          3) Compute head and hydropower rate per day for each scenario.
          4) Split timestep into month segments; optionally filter by month.
          5) Add energy_per_day * segment_days to the corresponding annual bucket.
        """
        ts = self.model.timestepper.current
        t_i = ts.index

        dt_start = self._dt_starts[t_i]
        dt_end_raw = dt_start + pd.Timedelta(days=float(ts.days))

        # Clip to model period to ensure we do not count beyond configured end date.
        dt_end = dt_end_raw if dt_end_raw <= self._model_stop else self._model_stop
        if dt_end <= dt_start:
            return

        n_scen = len(self.model.scenarios.combinations)

        # Model flow is a per-day rate in the model's flow units (your case: Mm³/day).
        q = np.asarray(self.node.flow, dtype=float)
        if q.shape == ():
            q = np.full(n_scen, float(q), dtype=float)

        # Compute head per scenario
        water_elev = self._water_elevation_values(n_scen)
        if water_elev is None:
            head = np.full(n_scen, self.turbine_elevation, dtype=float)
        else:
            head = water_elev - self.turbine_elevation
        head = np.maximum(head, 0.0)

        # Compute energy rate per day (units implied by energy_unit_conversion)
        energy_per_day = self._hydropower_rate(q, head)

        # Month filter, if provided
        months_set = None
        if self._monthly_seasonality is not None:
            months_set = set(int(m) for m in self._monthly_seasonality)

        # Split and allocate the timestep by month boundary to avoid season bias with 7D timesteps
        for seg_start, seg_end in self._iter_month_segments(dt_start, dt_end):
            if months_set is not None and seg_start.month not in months_set:
                continue

            seg_days = self._segment_days(seg_start, seg_end)
            if seg_days <= 0:
                continue

            # Allocate to the year of the segment start. (Month segments do not cross year boundaries.)
            year = int(seg_start.year)
            year_slot = self._year_to_slot.get(year, None)
            if year_slot is None:
                # Outside the configured year range; ignore
                continue

            self._annual_totals[year_slot, :] += energy_per_day * seg_days

    def finish(self):
        """
        Finalise outputs:
          - Apply `factor` if provided.
          - Build an annual DataFrame indexed by year-end (YYYY-12-31) timestamps.
        """
        super().finish()

        annual = self._annual_totals
        if self.factor is not None:
            annual = annual * float(self.factor)

        sc_index = self.model.scenarios.multiindex
        annual_dt_index = pd.to_datetime([f"{y}-12-31" for y in self._year_labels])
        self._annual_df = pd.DataFrame(annual, index=annual_dt_index, columns=sc_index)

    def to_dataframe(self):
        """Return annual totals as a DataFrame (years x scenarios)."""
        if self._annual_df is None:
            self.finish()
        return self._annual_df

    def values(self):
        """
        Return a 1D array (n_scen,) by aggregating annual totals over years using `temporal_agg_func`.
        """
        if self._annual_df is None:
            self.finish()

        return self._temporal_aggregator.aggregate_2d(
            self._annual_df.values,
            axis=0,  # aggregate over years
            ignore_nan=self.ignore_nan,
        )

    @classmethod
    def load(cls, model, data):
        """
        Load the recorder from JSON.

        Supported keys (subset; standard Recorder keys also allowed):
          - node (required)
          - monthly_seasonality (optional)
          - water_elevation_parameter (optional)
          - turbine_elevation, efficiency, density, flow_unit_conversion, energy_unit_conversion (optional)
          - factor (optional; consumed by this class and applied in finish())
          - temporal_agg_func (optional)
        """
        node = model._get_node_from_ref(model, data.pop("node"))
        monthly_seasonality = data.pop("monthly_seasonality", None)

        wep_data = data.pop("water_elevation_parameter", None)
        water_elevation_parameter = load_parameter(model, wep_data) if wep_data is not None else None

        return cls(
            model,
            node,
            monthly_seasonality=monthly_seasonality,
            water_elevation_parameter=water_elevation_parameter,
            **data,
        )


AnnualHydroPowerRecorder.register()



# =============================================================================
# Annual deficit / robustness recorders (run-length family)
# =============================================================================

class _BaseAnnualDeficitFlagRecorder(NodeRecorder):
    """
    Base class for annual deficit-flag recorders.

    This base implements the shared mechanics used by the annual deficit/run-length recorders:

    - Build the simulation-year index strictly from the model timestepper definition (start/end years).
    - During model execution (`after()`), compute a per-timestep, per-scenario boolean "deficit" mask.
    - Allocate any deficit occurrence to the appropriate simulation year(s), correctly handling:
        * non-monthly/non-annual timesteps (e.g., 7D); and
        * optional month filtering (`monthly_seasonality`) without pandas resampling artefacts.
    - Provide `self._annual_deficit` with shape (n_years, n_scen) of boolean deficit-year flags.

    Subclasses only need to define `_deficit_mask()` (per timestep, per scenario) and may implement
    any scalar metric derived from the annual deficit sequence (e.g., deficit-year frequency, episode
    count, maximum consecutive deficit years).

    Failure / deficit semantics
    ---------------------------
    This base does not prescribe the deficit definition; subclasses define it. Two common definitions are:
      - Flow deficit: flow < threshold * max_flow (for demand nodes / river gauges / supply points)
      - Storage deficit: volume < threshold * max_volume (for reservoirs)

    Model period handling (no extra "zero year")
    --------------------------------------------
    Annual indexing is capped to:
        start.year .. end.year (inclusive)
    where start/end are taken from the JSON timestepper configuration when available.

    Optional month filtering
    ------------------------
    If `monthly_seasonality` is provided (list of months 1..12), only the portion of each timestep that
    overlaps those months is considered when flagging annual deficits. This supports irrigation seasons
    or seasonal performance assessment.

    Notes on timestep allocation
    ----------------------------
    Using pandas `resample('Y')` or `resample('M')` on 7D timesteps can misattribute events at period
    boundaries (e.g., a timestep starting on Jan 28 spans into February). This base avoids those artefacts
    by splitting each timestep at calendar boundaries and allocating by exact overlap.

    """

    def __init__(self, model, node, threshold, monthly_seasonality=None, **kwargs):
        super().__init__(model, node, **kwargs)

        self.threshold = float(threshold)
        self._monthly_seasonality = monthly_seasonality

        self._dt_starts = None
        self._model_start = None
        self._model_end = None
        self._model_stop = None

        self._year_labels = None
        self._year_to_slot = None

        # Boolean annual deficit flags (n_years, n_scen)
        self._annual_deficit = None

        # Cache for derived metrics
        self._values_cache = None

    def setup(self):
        super().setup()

        dt_index = self.model.timestepper.datetime_index
        self._dt_starts = _as_timestamp_index(dt_index)

        self._model_start, self._model_end = _model_start_end(self.model, self._dt_starts)
        self._model_stop = self._model_end + pd.Timedelta(days=1)

        y0 = int(self._model_start.year)
        y1 = int(self._model_end.year)

        self._year_labels = np.arange(y0, y1 + 1, dtype=int)
        self._year_to_slot = {y: i for i, y in enumerate(self._year_labels)}

        n_years = len(self._year_labels)
        n_scen = len(self.model.scenarios.combinations)

        self._annual_deficit = np.zeros((n_years, n_scen), dtype=bool)
        self._values_cache = None

    def reset(self):
        super().reset()
        self._annual_deficit[:, :] = False
        self._values_cache = None

    # ---- Calendar segmentation helpers (local to this base) -----------------

    def _iter_month_segments(self, dt_start, dt_end):
        yield from _iter_month_segments(dt_start, dt_end)

    def _iter_year_segments(self, dt_start, dt_end):
        yield from _iter_year_segments(dt_start, dt_end)

    def _deficit_mask(self):
        """
        Return a boolean array (n_scen,) for the *current timestep*, indicating which scenarios are in deficit.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    def after(self):
        """
        Update annual deficit flags based on the current timestep.

        A year is marked as a "deficit year" for a scenario if *any* deficit occurs in that year, after
        optional month filtering.
        """
        ts = self.model.timestepper.current
        dt_start = self._dt_starts[ts.index]

        dt_end_raw = dt_start + pd.Timedelta(days=float(ts.days))
        dt_end = dt_end_raw if dt_end_raw <= self._model_stop else self._model_stop
        if dt_end <= dt_start:
            return 0

        mask = self._deficit_mask()
        if not np.any(mask):
            return 0

        months_set = None
        if self._monthly_seasonality is not None:
            months_set = set(int(m) for m in self._monthly_seasonality)

        # If we are filtering months, split at month boundaries; otherwise split at year boundaries.
        if months_set is not None:
            segments = self._iter_month_segments(dt_start, dt_end)
        else:
            segments = self._iter_year_segments(dt_start, dt_end)

        for seg_start, seg_end in segments:
            if months_set is not None and seg_start.month not in months_set:
                continue
            if _segment_days(seg_start, seg_end) <= 0.0:
                continue

            year_slot = self._year_to_slot.get(int(seg_start.year), None)
            if year_slot is None:
                continue

            self._annual_deficit[year_slot, mask] = True

        return 0

    # ---- Audit helpers ------------------------------------------------------

    def _year_end_index(self):
        return pd.to_datetime([f"{y}-12-31" for y in self._year_labels])

    def _episode_id_matrix(self):
        """Return episode IDs (same shape as annual_deficit), with 0 for non-deficit years."""
        flags = self._annual_deficit
        n_years, n_scen = flags.shape
        episode_id = np.zeros((n_years, n_scen), dtype=float)

        for s in range(n_scen):
            eid = 0
            in_run = False
            for y in range(n_years):
                if flags[y, s]:
                    if not in_run:
                        eid += 1
                        in_run = True
                    episode_id[y, s] = float(eid)
                else:
                    in_run = False

        return episode_id

    def _audit_dataframe(self, include_episode_id=False):
        """
        Build a MultiIndex-column DataFrame for auditing, optionally including episode IDs.

        If include_episode_id is False, returns only deficit_year flags (0/1).
        If include_episode_id is True, returns both deficit_year and episode_id in a metric-level column.
        """
        idx = self._year_end_index()
        sc_index = self.model.scenarios.multiindex

        deficit = self._annual_deficit.astype(float)

        if not include_episode_id:
            return pd.DataFrame(deficit, index=idx, columns=sc_index)

        episode_id = self._episode_id_matrix()

        if isinstance(sc_index, pd.MultiIndex):
            deficit_cols = [("deficit_year",) + tuple(c) for c in sc_index]
            eid_cols = [("episode_id",) + tuple(c) for c in sc_index]
            col_names = ["metric"] + list(sc_index.names)
        else:
            deficit_cols = [("deficit_year", c) for c in sc_index]
            eid_cols = [("episode_id", c) for c in sc_index]
            col_names = ["metric", "scenario"]

        cols = pd.MultiIndex.from_tuples(deficit_cols + eid_cols, names=col_names)
        data = np.hstack([deficit, episode_id])
        return pd.DataFrame(data, index=idx, columns=cols)


class _AnnualFlowDeficitFlagRecorder(_BaseAnnualDeficitFlagRecorder):
    """
    Annual deficit flag base for *flow* nodes.

    Deficit definition:
        flow < threshold * max_flow
    """

    def _deficit_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            max_flow = float(node.get_max_flow(si))
            mask[gid] = flow[gid] < (max_flow * self.threshold)
        return mask


class _AnnualStorageDeficitFlagRecorder(_BaseAnnualDeficitFlagRecorder):
    """
    Annual deficit flag base for *storage* nodes.

    Deficit definition:
        volume < threshold * max_volume
    """

    def _deficit_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        vol = np.asarray(node.volume, dtype=float)
        if vol.shape == ():
            vol = np.full(n_scen, float(vol), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            max_vol = float(node.get_max_volume(si))
            mask[gid] = vol[gid] < (max_vol * self.threshold)
        return mask


# -----------------------------------------------------------------------------
# Annual deficit-year frequency
# -----------------------------------------------------------------------------

class AnnualDeficitYearsRecorder(_AnnualFlowDeficitFlagRecorder):
    """
    Annual deficit-years frequency recorder (per scenario), based on timestep deficits.

    Metric family and intent
    ------------------------
    This is a robustness / persistence-oriented frequency indicator. It answers:
        "How many simulation years experienced at least one deficit event?"

    Annual deficit-year definition
    ------------------------------
    For each scenario and year y, we define:

        deficit_timestep = 1  if  flow < threshold * max_flow
                         = 0  otherwise

    A year is a "deficit year" if it contains one or more deficit timesteps (after optional month filtering).
    The annual flag is therefore binary (0/1).

    Temporal aggregation
    --------------------
    `values()` returns a scalar per scenario by applying a Pywr temporal aggregator across years.

    By default:
        temporal_agg_func = "COUNT_NONZERO"

    which yields the number of deficit years.

    JSON usage (general)
    --------------------
    {
      "Some node: Annual deficit years": {
        "type": "AnnualDeficitYearsRecorder",
        "node": "NODE_NAME",
        "threshold": 0.9,
        "temporal_agg_func": "COUNT_NONZERO",
        "agg_func": "mean",
        "monthly_seasonality": [4,5,6,7,8,9]
      }
    }

    Outputs
    -------
    - to_dataframe(): annual deficit flags (0/1) by year-end timestamp and scenario.
    - values(): per-scenario scalar (aggregated across years).
    - aggregated_value(): scenario-aggregated scalar (via Pywr agg_func).
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop("temporal_agg_func", "COUNT_NONZERO")
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)

        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

        self._temporal_agg_func = temporal_agg_func
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=False)

    def values(self):
        arr = self._annual_deficit.astype(float)  # (n_years, n_scen)
        return self._temporal_aggregator.aggregate_2d(arr, axis=0, ignore_nan=self.ignore_nan)

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


AnnualDeficitYearsRecorder.register()


# -----------------------------------------------------------------------------
# Annual deficit episode count (runs of length >= K)
# -----------------------------------------------------------------------------

class AnnualDeficitEpisodeCountRecorder(_AnnualFlowDeficitFlagRecorder):
    """
    Count of annual deficit episodes with length >= K (per scenario).

    Metric definition
    -----------------
    Let D_y ∈ {0,1} denote whether year y is a deficit year for a scenario.
    A deficit episode is a maximal contiguous block of years where D_y = 1.
    If an episode has length L_i (years), then for a chosen threshold K:

        EpisodeCount_K = #{ i : L_i >= K }

    Parameters
    ----------
    consecutive_years : int, default 2
        Minimum episode length (years) required to count an episode.

    JSON usage (general)
    --------------------
    {
      "Some node: Deficit episode count (K=2)": {
        "type": "AnnualDeficitEpisodeCountRecorder",
        "node": "NODE_NAME",
        "threshold": 0.9,
        "consecutive_years": 2,
        "agg_func": "mean"
      }
    }

    Notes
    -----
    This metric counts each qualifying episode once, regardless of how long it lasts beyond K.
    If you want longer episodes to contribute more, use `AnnualDeficitEpisodeExcessYearsRecorder`.
    """

    def __init__(self, model, node, threshold, consecutive_years=2, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

        self.consecutive_years = int(consecutive_years)
        if self.consecutive_years < 2:
            raise ValueError("consecutive_years must be >= 2")

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape
        K = self.consecutive_years

        out = np.zeros(n_scen, dtype=float)

        for s in range(n_scen):
            run = 0
            count = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                else:
                    if run >= K:
                        count += 1
                    run = 0
            if run >= K:
                count += 1
            out[s] = float(count)

        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        consecutive_years = data.pop("consecutive_years", 2)
        return cls(model, node, threshold, consecutive_years=consecutive_years, **data)


AnnualDeficitEpisodeCountRecorder.register()


# -----------------------------------------------------------------------------
# Annual deficit episode excess years (persistence-weighted)
# -----------------------------------------------------------------------------

class AnnualDeficitEpisodeExcessYearsRecorder(_AnnualFlowDeficitFlagRecorder):
    """
    Sum of excess years beyond a minimum consecutive deficit threshold K (per scenario).

    Metric definition
    -----------------
    For each deficit episode i with length L_i and threshold K:

        Excess_K = Σ_i max(0, L_i - K + 1)

    This can be interpreted as the number of length-K windows contained within deficit runs.

    Parameters
    ----------
    consecutive_years : int, default 2

    JSON usage (general)
    --------------------
    {
      "Some node: Deficit excess years (K=3)": {
        "type": "AnnualDeficitEpisodeExcessYearsRecorder",
        "node": "NODE_NAME",
        "threshold": 0.9,
        "consecutive_years": 3,
        "agg_func": "mean"
      }
    }
    """

    def __init__(self, model, node, threshold, consecutive_years=2, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

        self.consecutive_years = int(consecutive_years)
        if self.consecutive_years < 2:
            raise ValueError("consecutive_years must be >= 2")

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape
        K = self.consecutive_years

        out = np.zeros(n_scen, dtype=float)

        for s in range(n_scen):
            run = 0
            excess = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                else:
                    if run >= K:
                        excess += (run - K + 1)
                    run = 0
            if run >= K:
                excess += (run - K + 1)
            out[s] = float(excess)

        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        consecutive_years = data.pop("consecutive_years", 2)
        return cls(model, node, threshold, consecutive_years=consecutive_years, **data)


AnnualDeficitEpisodeExcessYearsRecorder.register()


# -----------------------------------------------------------------------------
# Annual maximum consecutive deficit years (worst-case persistence)
# -----------------------------------------------------------------------------

class AnnualMaxConsecutiveDeficitYearsRecorder(_AnnualFlowDeficitFlagRecorder):
    """
    Maximum consecutive annual deficit years (per scenario).

    Metric definition
    -----------------
    Let L_i be the length of each deficit episode i (contiguous deficit-year run).
    This recorder returns:

        MaxRun = max_i L_i

    JSON usage (general)
    --------------------
    {
      "Some node: Max consecutive deficit years": {
        "type": "AnnualMaxConsecutiveDeficitYearsRecorder",
        "node": "NODE_NAME",
        "threshold": 0.9,
        "agg_func": "mean"
      }
    }
    """

    def __init__(self, model, node, threshold, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape

        out = np.zeros(n_scen, dtype=float)

        for s in range(n_scen):
            run = 0
            max_run = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
            out[s] = float(max_run)

        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


AnnualMaxConsecutiveDeficitYearsRecorder.register()


# -----------------------------------------------------------------------------
# Storage-node equivalents (volume < threshold * max_volume)
# -----------------------------------------------------------------------------

class AnnualStorageDeficitYearsRecorder(_AnnualStorageDeficitFlagRecorder):
    """
    Storage-node equivalent of AnnualDeficitYearsRecorder.

    Deficit definition:
        volume < threshold * max_volume
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop("temporal_agg_func", "COUNT_NONZERO")
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)

        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

        self._temporal_agg_func = temporal_agg_func
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=False)

    def values(self):
        arr = self._annual_deficit.astype(float)
        return self._temporal_aggregator.aggregate_2d(arr, axis=0, ignore_nan=self.ignore_nan)

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


AnnualStorageDeficitYearsRecorder.register()


class AnnualStorageDeficitEpisodeCountRecorder(_AnnualStorageDeficitFlagRecorder):
    """Storage-node equivalent of AnnualDeficitEpisodeCountRecorder."""

    def __init__(self, model, node, threshold, consecutive_years=2, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)
        self.consecutive_years = int(consecutive_years)
        if self.consecutive_years < 2:
            raise ValueError("consecutive_years must be >= 2")

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape
        K = self.consecutive_years

        out = np.zeros(n_scen, dtype=float)
        for s in range(n_scen):
            run = 0
            count = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                else:
                    if run >= K:
                        count += 1
                    run = 0
            if run >= K:
                count += 1
            out[s] = float(count)
        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        consecutive_years = data.pop("consecutive_years", 2)
        return cls(model, node, threshold, consecutive_years=consecutive_years, **data)


AnnualStorageDeficitEpisodeCountRecorder.register()


class AnnualStorageDeficitEpisodeExcessYearsRecorder(_AnnualStorageDeficitFlagRecorder):
    """Storage-node equivalent of AnnualDeficitEpisodeExcessYearsRecorder."""

    def __init__(self, model, node, threshold, consecutive_years=2, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)
        self.consecutive_years = int(consecutive_years)
        if self.consecutive_years < 2:
            raise ValueError("consecutive_years must be >= 2")

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape
        K = self.consecutive_years

        out = np.zeros(n_scen, dtype=float)
        for s in range(n_scen):
            run = 0
            excess = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                else:
                    if run >= K:
                        excess += (run - K + 1)
                    run = 0
            if run >= K:
                excess += (run - K + 1)
            out[s] = float(excess)
        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        consecutive_years = data.pop("consecutive_years", 2)
        return cls(model, node, threshold, consecutive_years=consecutive_years, **data)


AnnualStorageDeficitEpisodeExcessYearsRecorder.register()


class AnnualMaxConsecutiveStorageDeficitYearsRecorder(_AnnualStorageDeficitFlagRecorder):
    """Storage-node equivalent of AnnualMaxConsecutiveDeficitYearsRecorder."""

    def __init__(self, model, node, threshold, **kwargs):
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, threshold, monthly_seasonality=monthly_seasonality, **kwargs)

    def to_dataframe(self):
        return self._audit_dataframe(include_episode_id=True)

    def values(self):
        flags = self._annual_deficit
        n_years, n_scen = flags.shape

        out = np.zeros(n_scen, dtype=float)
        for s in range(n_scen):
            run = 0
            max_run = 0
            for y in range(n_years):
                if flags[y, s]:
                    run += 1
                    if run > max_run:
                        max_run = run
                else:
                    run = 0
            out[s] = float(max_run)
        return out

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


AnnualMaxConsecutiveStorageDeficitYearsRecorder.register()



# =============================================================================
# Helpers
# =============================================================================

def _as_timestamp_index(dt_index):
    """Return a list[pd.Timestamp] from a Pywr datetime_index (PeriodIndex or datetime-like)."""
    if isinstance(dt_index, pd.PeriodIndex):
        dt_index = dt_index.to_timestamp()
    return [pd.Timestamp(d) for d in dt_index]


def _model_start_end(model, dt_starts):
    """
    Determine model start/end dates.

    Prefer timestepper.start/end if available (JSON timestepper definition),
    otherwise fall back to datetime_index bounds.
    """
    ts_start = getattr(model.timestepper, "start", None)
    ts_end = getattr(model.timestepper, "end", None)

    start = pd.Timestamp(ts_start) if ts_start is not None else dt_starts[0]
    end = pd.Timestamp(ts_end) if ts_end is not None else dt_starts[-1]
    return start, end


def _iter_month_segments(dt_start, dt_end):
    """
    Yield (seg_start, seg_end) segments that do not cross month boundaries.
    This is essential for correct month attribution with non-monthly timesteps (e.g., 7D).
    """
    cur = dt_start
    while cur < dt_end:
        next_month = (cur.to_period("M") + 1).to_timestamp()
        seg_end = next_month if next_month < dt_end else dt_end
        yield cur, seg_end
        cur = seg_end


def _iter_year_segments(dt_start, dt_end):
    """Yield (seg_start, seg_end) segments that do not cross year boundaries."""
    cur = dt_start
    while cur < dt_end:
        next_year = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        seg_end = next_year if next_year < dt_end else dt_end
        yield cur, seg_end
        cur = seg_end


def _segment_days(seg_start, seg_end):
    """Return segment duration in days (float)."""
    return (seg_end - seg_start).total_seconds() / 86400.0


def _scenario_aggregate_fallback(recorder, values_1d):
    """
    Fallback scenario aggregation for custom recorders if super().aggregated_value() is unavailable.
    """
    agg_name = getattr(recorder, "agg_func", None) or "mean"
    agg = Aggregator(agg_name)
    agg.func = agg_name
    return agg.aggregate_1d(np.asarray(values_1d, dtype=float), ignore_nan=recorder.ignore_nan)


class _BasePeriodFailureFlagRecorder(NodeRecorder):
    """
    Base recorder to compute boolean failure flags per calendar period (month or year), per scenario.

    Subclasses must implement:
      - _failure_mask() -> boolean array (n_scen,) for the *current timestep*
    """

    def __init__(self, model, node, period="M", monthly_seasonality=None, **kwargs):
        super().__init__(model, node, **kwargs)

        if period not in ("M", "Y"):
            raise ValueError("period must be 'M' (month) or 'Y' (year)")

        self.period = period
        self._monthly_seasonality = monthly_seasonality

        self._dt_starts = None
        self._model_start = None
        self._model_end = None
        self._model_stop = None  # exclusive end boundary (end + 1 day)

        self._period_index = None        # pd.PeriodIndex
        self._period_end_index = None    # pd.DatetimeIndex (period ends)
        self._period_to_slot = None      # dict key->slot
        self._failed = None              # bool (n_periods, n_scen)

    def setup(self):
        super().setup()

        self._dt_starts = _as_timestamp_index(self.model.timestepper.datetime_index)
        self._model_start, self._model_end = _model_start_end(self.model, self._dt_starts)
        self._model_stop = self._model_end + pd.Timedelta(days=1)

        # Build period index strictly from timestepper start/end (prevents “extra zero period”)
        if self.period == "M":
            self._period_index = pd.period_range(self._model_start, self._model_end, freq="M")
            # Use period end timestamps for reporting/auditing
            self._period_end_index = self._period_index.to_timestamp(how="end").normalize()
            self._period_to_slot = {(p.year, p.month): i for i, p in enumerate(self._period_index)}
        else:
            self._period_index = pd.period_range(self._model_start, self._model_end, freq="Y")
            self._period_end_index = self._period_index.to_timestamp(how="end").normalize()
            self._period_to_slot = {p.year: i for i, p in enumerate(self._period_index)}

        n_periods = len(self._period_index)
        n_scen = len(self.model.scenarios.combinations)
        self._failed = np.zeros((n_periods, n_scen), dtype=bool)

    def reset(self):
        super().reset()
        self._failed[:, :] = False

    def _iter_period_segments(self, dt_start, dt_end):
        if self.period == "M":
            yield from _iter_month_segments(dt_start, dt_end)
        else:
            yield from _iter_year_segments(dt_start, dt_end)

    def _segment_to_slot(self, seg_start):
        if self.period == "M":
            return self._period_to_slot.get((seg_start.year, seg_start.month), None)
        return self._period_to_slot.get(seg_start.year, None)

    def _failure_mask(self):
        raise NotImplementedError

    def after(self):
        """
        Update the per-period failure flags using the current timestep.

        Key behavior:
        - Timestep is clipped to model_stop = (end + 1 day), preventing accidental spill into an extra period.
        - If a timestep overlaps multiple months/years, we flag *each* overlapped period when failure occurs.
          This avoids boundary artefacts that appear when using pandas resampling on 7D timesteps.
        - If monthly_seasonality is provided, only segments whose month is in the set are considered.
        """
        ts = self.model.timestepper.current
        dt_start = self._dt_starts[ts.index]

        dt_end_raw = dt_start + pd.Timedelta(days=float(ts.days))
        dt_end = dt_end_raw if dt_end_raw <= self._model_stop else self._model_stop
        if dt_end <= dt_start:
            return 0

        mask = self._failure_mask()
        if not np.any(mask):
            return 0

        months_set = None
        if self._monthly_seasonality is not None:
            months_set = set(int(m) for m in self._monthly_seasonality)

        for seg_start, seg_end in self._iter_period_segments(dt_start, dt_end):
            if months_set is not None and seg_start.month not in months_set:
                continue
            if _segment_days(seg_start, seg_end) <= 0.0:
                continue

            slot = self._segment_to_slot(seg_start)
            if slot is None:
                continue

            self._failed[slot, mask] = True

        return 0

    def to_dataframe(self):
        """
        Return a DataFrame of period failure flags.

        Index:
          - period ends (month-end or year-end timestamps), capped to timestepper start/end.

        Columns:
          - model.scenarios.multiindex
        """
        sc_index = self.model.scenarios.multiindex
        return pd.DataFrame(self._failed.astype(float), index=self._period_end_index, columns=sc_index)

    def values(self):
        """
        Default: time-based reliability over periods.

            reliability = 1 - (#failed periods / #periods)

        Subclasses may override if they represent a different metric.
        """
        n = self._failed.shape[0]
        if n == 0:
            return np.zeros(len(self.model.scenarios.combinations), dtype=float)
        return 1.0 - (self._failed.sum(axis=0).astype(float) / float(n))

    def aggregated_value(self):
        """Scenario aggregation using Pywr's agg_func if available; otherwise fallback."""
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())


# =============================================================================
# Storage-based reliability (replaces ReservoirMonthly/AnnualReliabilityRecorder)
# =============================================================================

class MonthlyStorageThresholdReliabilityRecorder(_BasePeriodFailureFlagRecorder):
    """
    Monthly storage-threshold reliability (per scenario).

    Metric family
    -------------
    This is a time-based reliability metric within the Reliability–Resilience–Vulnerability (RRV) family,
    where "failure" is defined as storage falling below a threshold fraction of maximum storage. The
    reliability is the fraction of months without failure. Hashimoto et al. (1982) provides the classic
    framing of time-based reliability. :contentReference[oaicite:10]{index=10}

    Definition
    ----------
    For each scenario and month m:

        F_m = 1 if storage is below threshold at any timestep overlapping month m
            = 0 otherwise

    Then:

        Reliability_monthly = 1 - (Σ_m F_m / N_months)

    Parameters
    ----------
    threshold : float
        Failure threshold as a fraction of max volume:
            failure if volume < threshold * max_volume
    monthly_seasonality : list[int], optional
        If provided, only months in this list contribute to failure counting.

    JSON usage (general)
    --------------------
    {
      "Some reservoir: Monthly storage reliability": {
        "type": "MonthlyStorageThresholdReliabilityRecorder",
        "node": "RESERVOIR_NODE_NAME",
        "threshold": 0.2,
        "agg_func": "mean",
        "monthly_seasonality": [4,5,6,7,8,9]
      }
    }
    """

    def __init__(self, model, node, threshold, **kwargs):
        self.threshold = float(threshold)
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, period="M", monthly_seasonality=monthly_seasonality, **kwargs)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        vol = np.asarray(node.volume, dtype=float)
        if vol.shape == ():
            vol = np.full(n_scen, float(vol), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            max_vol = float(node.get_max_volume(si))
            mask[gid] = vol[gid] < (max_vol * self.threshold)
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


MonthlyStorageThresholdReliabilityRecorder.register()


class AnnualStorageThresholdReliabilityRecorder(_BasePeriodFailureFlagRecorder):
    """
    Annual storage-threshold reliability (per scenario).

    This is the same reliability concept as MonthlyStorageThresholdReliabilityRecorder, but at the year scale:

        Reliability_annual = 1 - (#years with any failure / #years)

    It is often used alongside monthly reliability to distinguish short transient failures from persistent
    annual shortages.

    See Hashimoto et al. (1982) for the RRV framework. :contentReference[oaicite:11]{index=11}
    """

    def __init__(self, model, node, threshold, **kwargs):
        self.threshold = float(threshold)
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, period="Y", monthly_seasonality=monthly_seasonality, **kwargs)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        vol = np.asarray(node.volume, dtype=float)
        if vol.shape == ():
            vol = np.full(n_scen, float(vol), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            max_vol = float(node.get_max_volume(si))
            mask[gid] = vol[gid] < (max_vol * self.threshold)
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


AnnualStorageThresholdReliabilityRecorder.register()


# =============================================================================
# Supply-based monthly reliability (replaces SupplyReliabilityRecorder)
# =============================================================================

class MonthlyDemandSatisfactionReliabilityRecorder(_BasePeriodFailureFlagRecorder):
    """
    Monthly demand-satisfaction reliability (per scenario).

    Metric family
    -------------
    This is a time-based reliability metric in the RRV family (Hashimoto et al., 1982). :contentReference[oaicite:12]{index=12}
    It measures how frequently monthly demand satisfaction meets a target, where "failure" is defined as a
    relative deficit exceeding a threshold.

    Definition
    ----------
    At each timestep:

        demand = node.get_max_flow(scenario)
        supply = node.flow[scenario]

        deficit_fraction = 0                           if demand == 0
                         = (demand - supply) / demand  otherwise

    For each month m:

        F_m = 1 if deficit_fraction > deficit_threshold at any timestep overlapping month m
            = 0 otherwise

    Then:

        Reliability_monthly = 1 - (Σ_m F_m / N_months)

    Parameters
    ----------
    deficit_threshold : float, default 0.01
        Relative deficit fraction above which the month is considered failed.
    monthly_seasonality : list[int], optional
        If provided, evaluate only those months.

    Notes on demand=0 months
    ------------------------
    Months/timesteps with demand==0 are treated as "no deficit" (deficit_fraction = 0), consistent with
    intermittent irrigation seasons or crop calendars.

    JSON usage (general)
    --------------------
    {
      "Some demand node: Monthly reliability": {
        "type": "MonthlyDemandSatisfactionReliabilityRecorder",
        "node": "DEMAND_NODE_NAME",
        "deficit_threshold": 0.01,
        "agg_func": "mean",
        "monthly_seasonality": [4,5,6,7,8,9]
      }
    }
    """

    def __init__(self, model, node, deficit_threshold=0.01, **kwargs):
        self.deficit_threshold = float(deficit_threshold)
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, period="M", monthly_seasonality=monthly_seasonality, **kwargs)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            demand = float(node.get_max_flow(si))
            if demand <= 0.0:
                deficit_fraction = 0.0
            else:
                deficit_fraction = (demand - flow[gid]) / demand
            mask[gid] = deficit_fraction > self.deficit_threshold
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        deficit_threshold = data.pop("deficit_threshold", 0.01)
        return cls(model, node, deficit_threshold=deficit_threshold, **data)


MonthlyDemandSatisfactionReliabilityRecorder.register()

# -----------------------------------------------------------------------------
# Demand-satisfaction reliability (annual) and resilience (monthly/annual)
# -----------------------------------------------------------------------------

class AnnualDemandSatisfactionReliabilityRecorder(_BasePeriodFailureFlagRecorder):
    """
    Annual demand-satisfaction reliability recorder (per scenario).

    This recorder flags each simulation year as failed if the node demand satisfaction falls below a threshold
    at least once within that year. Demand satisfaction is evaluated at the model timestep resolution and then
    aggregated to annual periods using exact boundary handling (consistent with the base period recorder).

    Failure condition (per timestep)
    --------------------------------
    Let:
        demand = node.get_max_flow(scenario)
        supply = node.flow[scenario]
    Then the demand satisfaction ratio is:
        r = supply / demand

    A timestep is a "failure" if:
        r < (1 - deficit_threshold)

    Special cases:
    - demand == 0: the timestep is treated as non-failing (no required supply).

    Metric
    ------
    values() returns the fraction of non-failing years (i.e., reliability as a probability in [0,1]):
        Reliability = 1 - mean(failure_year_flag)

    where failure_year_flag is 1 if any failure occurs in that year, else 0.

    JSON usage (general)
    --------------------
    {
      "Some node: Annual demand satisfaction reliability": {
        "type": "AnnualDemandSatisfactionReliabilityRecorder",
        "node": "NODE_NAME",
        "deficit_threshold": 0.01,
        "agg_func": "mean"
      }
    }

    Notes
    -----
    - This is conceptually aligned with the reliability definition in the RRV family (e.g., Hashimoto et al., 1982),
      but the specific failure definition here is based on demand satisfaction.
    """

    def __init__(self, model, node, deficit_threshold=0.01, **kwargs):
        super().__init__(model, node, period="Y", **kwargs)
        self.deficit_threshold = float(deficit_threshold)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            demand = float(node.get_max_flow(si))
            if demand <= 0.0:
                mask[gid] = False
            else:
                r = flow[gid] / demand
                mask[gid] = r < (1.0 - self.deficit_threshold)
        return mask

    def values(self):
        # failure matrix is built at the chosen period scale (Y)
        f = self._failed.astype(int)  # (T_years, n_scen)
        if f.shape[0] == 0:
            return np.zeros(f.shape[1], dtype=float)
        return 1.0 - np.mean(f, axis=0)

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        deficit_threshold = data.pop("deficit_threshold", 0.01)
        return cls(model, node, deficit_threshold=deficit_threshold, **data)


AnnualDemandSatisfactionReliabilityRecorder.register()





# =============================================================================
# Annual deficit fraction (replaces AnnualDeficitRecorder)
# =============================================================================

class AnnualDemandDeficitFractionRecorder(NodeRecorder):
    """
    Annual demand deficit fraction recorder (per scenario), with Pywr-style temporal aggregation.

    Metric family
    -------------
    This is a magnitude-based deficit metric often used alongside time-based reliability. It is related to
    vulnerability concepts because it captures the extent of shortfall (but it is normalised here).

    Definition
    ----------
    For each year y and scenario s:

        S_y = total supplied volume in year y
        D_y = total demanded volume in year y

        deficit_fraction_y = 0                    if D_y == 0
                           = 1 - (S_y / D_y)      otherwise

    The recorder supports a temporal aggregation over years using Pywr's Aggregator:
      - mean, max, min, median, etc.

    IMPORTANT: flow unit handling
    -----------------------------
    Many Pywr models treat node.flow as a per-timestep volume. However, you indicated your models can use
    flow units like m³/day (a rate) with a 7D timestep. In that case you must integrate rate × duration.

    This recorder supports both cases using `flow_is_rate`:

      - flow_is_rate = False (default):
          node.flow and node.get_max_flow are treated as per-timestep volumes.
          If a timestep crosses a year boundary, volume is allocated proportionally by days.

      - flow_is_rate = True:
          node.flow and node.get_max_flow are treated as rates per day.
          Volume contribution = rate × segment_days.

    Parameters
    ----------
    temporal_agg_func : str, default "mean"
        Pywr Aggregator function applied across annual deficit fractions.
    flow_is_rate : bool, default False
        See unit handling notes above.

    JSON usage (general)
    --------------------
    {
      "Some demand node: Annual deficit fraction": {
        "type": "AnnualDemandDeficitFractionRecorder",
        "node": "DEMAND_NODE_NAME",
        "temporal_agg_func": "mean",
        "flow_is_rate": true,
        "agg_func": "mean"
      }
    }
    """

    def __init__(self, model, node, temporal_agg_func="mean", flow_is_rate=False, **kwargs):
        super().__init__(model, node, **kwargs)
        self.flow_is_rate = bool(flow_is_rate)

        self._temporal_agg_func = temporal_agg_func
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self._temporal_aggregator.func = temporal_agg_func

        self._dt_starts = None
        self._model_start = None
        self._model_end = None
        self._model_stop = None

        self._years = None
        self._year_to_slot = None

        self._supply = None  # (n_years, n_scen) annual volume
        self._demand = None  # (n_years, n_scen) annual volume

    def setup(self):
        super().setup()

        self._dt_starts = _as_timestamp_index(self.model.timestepper.datetime_index)
        self._model_start, self._model_end = _model_start_end(self.model, self._dt_starts)
        self._model_stop = self._model_end + pd.Timedelta(days=1)

        y0 = int(self._model_start.year)
        y1 = int(self._model_end.year)
        self._years = np.arange(y0, y1 + 1, dtype=int)
        self._year_to_slot = {y: i for i, y in enumerate(self._years)}

        n_years = len(self._years)
        n_scen = len(self.model.scenarios.combinations)
        self._supply = np.zeros((n_years, n_scen), dtype=float)
        self._demand = np.zeros((n_years, n_scen), dtype=float)

    def reset(self):
        super().reset()
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        dt_start = self._dt_starts[ts.index]

        dt_end_raw = dt_start + pd.Timedelta(days=float(ts.days))
        dt_end = dt_end_raw if dt_end_raw <= self._model_stop else self._model_stop
        if dt_end <= dt_start:
            return 0

        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        # Allocate contributions by year segments
        for seg_start, seg_end in _iter_year_segments(dt_start, dt_end):
            seg_days = _segment_days(seg_start, seg_end)
            if seg_days <= 0.0:
                continue

            year_slot = self._year_to_slot.get(int(seg_start.year), None)
            if year_slot is None:
                continue

            if self.flow_is_rate:
                w = seg_days  # rate per day -> volume
            else:
                # per-timestep volume -> allocate proportionally by duration
                w = seg_days / float(ts.days)

            for si in scen:
                gid = si.global_id
                demand = float(node.get_max_flow(si))
                self._supply[year_slot, gid] += flow[gid] * w
                self._demand[year_slot, gid] += demand * w

        return 0

    def to_dataframe(self):
        """Annual deficit fraction time series per scenario."""
        idx = pd.to_datetime([f"{y}-12-31" for y in self._years])
        sc_index = self.model.scenarios.multiindex

        supply = self._supply
        demand = self._demand

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = 1.0 - (supply / demand)
        frac = np.where(demand <= 0.0, 0.0, frac)

        return pd.DataFrame(frac, index=idx, columns=sc_index)

    def values(self):
        """Return per-scenario aggregated annual deficit fraction."""
        df = self.to_dataframe()
        arr = df.values  # (n_years, n_scen)
        return self._temporal_aggregator.aggregate_2d(arr, axis=0, ignore_nan=self.ignore_nan)

    def aggregated_value(self):
        try:
            return super().aggregated_value()
        except Exception:
            return _scenario_aggregate_fallback(self, self.values())

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        temporal_agg_func = data.pop("temporal_agg_func", "mean")
        flow_is_rate = data.pop("flow_is_rate", False)
        return cls(model, node, temporal_agg_func=temporal_agg_func, flow_is_rate=flow_is_rate, **data)


AnnualDemandDeficitFractionRecorder.register()


# =============================================================================
# Resilience (classic recovery probability) for period failure series
# =============================================================================

class PeriodFailureResilienceRecorder(_BasePeriodFailureFlagRecorder):
    """
    Resilience as probability of recovery from failure (per scenario), computed from period failure flags.

    Metric family
    -------------
    This aligns with the classic RRV “resiliency/resilience” definition:
      resilience ≈ probability of returning to satisfactory state immediately after a failure period.
    See Hashimoto et al. (1982). :contentReference[oaicite:13]{index=13}

    Definition
    ----------
    Given a period failure series F_t ∈ {0,1} (per scenario), with t=1..T:

        recoveries = count of transitions where F_{t-1}=1 and F_t=0
        failures    = count of periods where F_{t}=1  (or equivalently, count of failure “states”)

        resilience = recoveries / failures   if failures > 0
                   = 1.0                    if failures == 0  (never failed)

    Subclasses define the failure condition (storage threshold or demand deficit). This class only changes
    how `values()` are computed from the already-built period failure matrix.

    Notes
    -----
    - This is complementary to reliability.
    - If you need duration-based resilience, see the duration-based recorders in literature. :contentReference[oaicite:14]{index=14}
    """

    def values(self):
        f = self._failed.astype(int)  # (T, n_scen)
        if f.shape[0] == 0:
            return np.zeros(f.shape[1], dtype=float)

        failures = f.sum(axis=0).astype(float)
        transitions = ((f[:-1, :] == 1) & (f[1:, :] == 0)).sum(axis=0).astype(float)

        out = np.ones_like(failures, dtype=float)
        mask = failures > 0.0
        out[mask] = transitions[mask] / failures[mask]
        return out



# -----------------------------------------------------------------------------
# Demand-satisfaction resilience (period recovery probability)
# -----------------------------------------------------------------------------

class MonthlyDemandSatisfactionResilienceRecorder(PeriodFailureResilienceRecorder):
    """
    Monthly demand-satisfaction resilience recorder (per scenario).

    Resilience is estimated as the conditional probability of recovery given a failure in the previous period:
        Resilience = P(non-failure at t | failure at t-1)

    This recorder uses the same monthly failure definition as `MonthlyDemandSatisfactionReliabilityRecorder`
    (i.e., demand satisfaction falling below 1 - deficit_threshold).

    JSON usage (general)
    --------------------
    {
      "Some node: Monthly demand satisfaction resilience": {
        "type": "MonthlyDemandSatisfactionResilienceRecorder",
        "node": "NODE_NAME",
        "deficit_threshold": 0.01,
        "agg_func": "mean"
      }
    }
    """

    def __init__(self, model, node, deficit_threshold=0.01, **kwargs):
        super().__init__(model, node, period="M", **kwargs)
        self.deficit_threshold = float(deficit_threshold)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            demand = float(node.get_max_flow(si))
            if demand <= 0.0:
                mask[gid] = False
            else:
                r = flow[gid] / demand
                mask[gid] = r < (1.0 - self.deficit_threshold)
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        deficit_threshold = data.pop("deficit_threshold", 0.01)
        return cls(model, node, deficit_threshold=deficit_threshold, **data)


MonthlyDemandSatisfactionResilienceRecorder.register()


class AnnualDemandSatisfactionResilienceRecorder(PeriodFailureResilienceRecorder):
    """
    Annual demand-satisfaction resilience recorder (per scenario).

    Same definition as `MonthlyDemandSatisfactionResilienceRecorder`, but evaluated at annual periods.
    """

    def __init__(self, model, node, deficit_threshold=0.01, **kwargs):
        super().__init__(model, node, period="Y", **kwargs)
        self.deficit_threshold = float(deficit_threshold)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        flow = np.asarray(node.flow, dtype=float)
        if flow.shape == ():
            flow = np.full(n_scen, float(flow), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            demand = float(node.get_max_flow(si))
            if demand <= 0.0:
                mask[gid] = False
            else:
                r = flow[gid] / demand
                mask[gid] = r < (1.0 - self.deficit_threshold)
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        deficit_threshold = data.pop("deficit_threshold", 0.01)
        return cls(model, node, deficit_threshold=deficit_threshold, **data)


AnnualDemandSatisfactionResilienceRecorder.register()


class MonthlyStorageThresholdResilienceRecorder(PeriodFailureResilienceRecorder):
    """
    Monthly storage-threshold resilience (recovery probability).

    Failure definition:
      volume < threshold * max_volume

    Period definition:
      monthly (calendar months, robustly split across month boundaries for 7D timesteps)

    JSON usage (general)
    --------------------
    {
      "Some reservoir: Monthly storage resilience": {
        "type": "MonthlyStorageThresholdResilienceRecorder",
        "node": "RESERVOIR_NODE_NAME",
        "threshold": 0.2,
        "agg_func": "mean"
      }
    }
    """

    def __init__(self, model, node, threshold, **kwargs):
        self.threshold = float(threshold)
        monthly_seasonality = kwargs.pop("monthly_seasonality", None)
        super().__init__(model, node, period="M", monthly_seasonality=monthly_seasonality, **kwargs)

    def _failure_mask(self):
        node = self.node
        scen = self.model.scenarios.combinations
        n_scen = len(scen)

        vol = np.asarray(node.volume, dtype=float)
        if vol.shape == ():
            vol = np.full(n_scen, float(vol), dtype=float)

        mask = np.zeros(n_scen, dtype=bool)
        for si in scen:
            gid = si.global_id
            max_vol = float(node.get_max_volume(si))
            mask[gid] = vol[gid] < (max_vol * self.threshold)
        return mask

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        threshold = data.pop("threshold")
        return cls(model, node, threshold, **data)


MonthlyStorageThresholdResilienceRecorder.register()