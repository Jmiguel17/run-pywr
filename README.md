# run-pywr

`run-pywr` is a Python package that provides a command-line interface (CLI) for running [Pywr](https://pywr.github.io/) models. It includes extended functionalities for various simulation scenarios, multi-objective optimization, and custom outputs tailored for specific project needs, such as those for The World Bank.

This package also defines a rich set of custom Pywr `Parameters` and `Recorders` to model complex water resource system behaviors and capture detailed performance metrics.

## Installation

To install the package, clone the repository and install it in editable mode using pip from the root directory:

```bash
git clone <repository-url>
cd run-pywr
pip install -e .
```

This will install the package in editable mode, meaning that any changes you make to the source code will be immediately reflected when you run the `run-pywr` command.

### Dependencies

The package dependencies are listed in the `pyproject.toml` file. You can install them directly using pip:

```bash
pip install -r requirements.txt
```

## Usage

The primary entry point for this package is the `run-pywr` command-line tool. You can see a list of all available commands by running:

```bash
run-pywr --help
```

### CLI Commands

The following commands are available to execute Pywr models:

---

#### `run`

Runs a standard Pywr model simulation.

**Usage:**
```bash
run-pywr run <FILENAME>
```
-   `<FILENAME>`: Path to the Pywr model JSON file.

**Description:**
This command loads and runs a Pywr model. It saves the following outputs to a directory named `<model_name>/outputs/`:
-   `*_parameters.h5`: A table of model parameters.
-   `*_nodes.csv`: A CSV file of node data.
-   `*_metrics.xlsx`: An Excel file containing recorder values and aggregated values.
-   `*_recorders.h5` or `*_recorders.csv`: A file containing detailed data from all recorders.

---

#### `run_simulation`

Runs a Pywr model and saves extended metrics, typically used for World Bank projects.

**Usage:**
```bash
run-pywr run_simulation <FILENAME>
```
-   `<FILENAME>`: Path to the Pywr model JSON file.

**Description:**
This command is tailored for detailed simulation analysis. It saves outputs into HDF5 stores, separating metrics, recorders, and aggregated data. It performs monthly resampling on recorder data for standardized reporting.

-   `*_metrics.h5`: Stores reliability, resilience, deficit, and other high-level metrics.
-   `*_recorders.h5`: Stores detailed, resampled time-series data from recorders.
-   `*_aggregated.h5`: Stores aggregated recorder values.

---

#### `run_dams_value`

Runs a Pywr model iteratively for the Amu Darya project, modifying the model to simulate the absence of specific dams.

**Usage:**
```bash
run-pywr run_dams_value <FILENAME>
```
-   `<FILENAME>`: Path to the Pywr model JSON file.

**Description:**
This command iterates through a predefined dictionary of dams. In each iteration, it modifies the input model by setting the storage and flows of specific dams to zero, effectively "removing" them from the simulation. It then runs the modified model and saves the results, allowing for an assessment of each dam's contribution or value to the system.

---

#### `run_scenarios`

Runs a Pywr model multiple times for different scenario slices.

**Usage:**
```bash
run-pywr run_scenarios [OPTIONS] <FILENAME>
```
-   `--start <INTEGER>`: The starting index for the scenario slice.
-   `--end <INTEGER>`: The ending index for the scenario slice.
-   `<FILENAME>`: Path to the Pywr model JSON file.

**Description:**
This command facilitates running a model across a range of scenarios. For each integer `i` from `start` to `end-1`, it modifies the model's scenario configuration to use `slice(i, i+1)` and runs a simulation. Outputs for each run are saved in a separate directory (`outputs_i_i+1`).

---

#### `search`

Performs a multi-objective search using the Platypus optimization framework.

**Usage:**
```bash
run-pywr search [OPTIONS] <FILENAME>
```
**Options:**
-   `--algorithm [NSGAII|NSGAIII|EpsMOEA|EpsNSGAII]`: The optimization algorithm to use.
-   `--max-nfe <INTEGER>`: Maximum number of function evaluations.
-   `--pop-size <INTEGER>`: Population size for the algorithm.
-   `--use-mpi`: Enable MPI for parallel processing.
-   `--seed <INTEGER>`: Random seed for reproducibility.
-   ...and other algorithm-specific parameters.

---

#### `pyborg`

Performs a multi-objective search using the Borg MOEA.

**Usage:**
```bash
run-pywr pyborg [OPTIONS] <FILENAME>
```
**Options:**
-   `--max-nfe <INTEGER>`: Maximum number of function evaluations.
-   `--use-mpi`: Enable MPI for parallel processing.
-   `--seed <INTEGER>`: Random seed.
-   ...and other Borg-specific parameters.

---

#### `pywr_mpi_borg`

Runs a multi-objective search using Borg with MPI, configured via a JSON file.

**Usage:**
```bash
run-pywr pywr_mpi_borg [OPTIONS] <CONFIG_FILE>
```
-   `<CONFIG_FILE>`: A JSON file containing the search configuration.

## Custom Components

This package includes custom parameters and recorders to extend Pywr's modeling capabilities.

### Custom Parameters

-   **`RectifierParameter`**: A parameter that returns its input value if it is within bounds `[0, inf]`, otherwise returns 0.
-   **`IndexVariableParameter`**: An `IndexParameter` that can be treated as a decision variable in an optimization.
-   **`IrrigationWaterRequirementParameter`**: A detailed model for calculating irrigation water demand based on rainfall, evapotranspiration (ET), crop factors, and efficiencies.
-   **`TransientDecisionParameter`**: Returns one of two values based on whether the current timestep is before or after a specified decision date. Can be used to model discrete events.
-   **`Domestic_deamnd_projection_parameter`**: Projects domestic water demand based on an annual percentage increase.
-   **`Demand_informed_release_Driekoppies`**: A parameter to calculate the release from Driekoppies Dam based on the storage levels of both Maguga and Driekoppies dams and downstream demands.
-   **`Demand_informed_release_Maguga`**: Calculates the release from Maguga Dam, considering downstream demands, hydropower requirements, and coordinated operation with Driekoppies Dam.
-   **`High_assurance_level` / `Low_assurance_level`**: Parameters designed to track demand satisfaction for nodes with different supply assurance levels.
-   **`Simple_Irr_demand_calculator_with_file`**: Calculates irrigation demand based on various factors read from external files.

### Custom Recorders

-   **`RollingNodeFlowRecorder`**: Records the mean flow of a node over a rolling window of N previous timesteps.
-   **`NumpyArrayAnnualNodeDeficitFrequencyRecorder`**: Records the number of years with at least one deficit event.
-   **Comparison Recorders**: A suite of recorders for comparing simulated results against observed data:
    -   `RootMeanSquaredErrorNodeRecorder`
    -   `NashSutcliffeEfficiencyNodeRecorder`
    -   `PercentBiasNodeRecorder`
    -   `NashSutcliffeEfficiencyStorageRecorder`
-   **Reliability and Resilience Recorders**:
    -   `ReservoirMonthlyReliabilityRecorder`: Calculates reliability based on the number of months a reservoir's volume is above a minimum threshold.
    -   `ReservoirAnnualReliabilityRecorder`: Calculates annual reliability for a reservoir.
    -   `SupplyReliabilityRecorder`: Calculates supply reliability for a demand node.
    -   `ReservoirResilienceRecorder`: Measures the average duration of deficit periods for a reservoir.
-   **Agricultural Recorders**:
    -   `AnnualDeficitRecorder`: Records the mean, max, or min annual supply deficit for a node.
    -   `AverageAnnualCropYieldScenarioRecorder`: Computes the average annual crop yield based on water supply.
    -   `TotalAnnualCropYieldScenarioRecorder`: Computes the total potential annual crop yield assuming no water limitations.
    -   `IrrigationSupplyReliabilityScenarioRecorder`: Calculates supply reliability specifically for irrigation, focusing on high-demand months.
    -   `CropCurtailmentRatioScenarioRecorder`: Records the annual curtailment ratio (supply/demand).
    -   `AnnualIrrigationSupplyReliabilityScenarioRecorder`: Calculates annual irrigation reliability based on a total volume threshold.
    -   `AverageAnnualIrrigationRevenueScenarioRecorder`: Calculates the average annual revenue from irrigation based on crop yield and price.
-   **Flow and Volume Recorders**:
    -   `AnnualSeasonalAccumulatedFlowRecorder`: Records the total annual flow accumulated only during user-specified months.
    -   `AnnualSeasonalVolumeRecorder`: Records the average annual volume during user-specified months.
-   **Hydropower and Constraint Recorders**:
    -   `AnnualHydropowerRecorder`: Calculates annual energy production from a hydropower node.
    -   `SeasonalTransferConstraintRecorder`: Records the performance of a seasonal flow transfer constraint.
