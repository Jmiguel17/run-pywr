import os
import sys
import json
import click
import tables
import random
import warnings

import numpy as np
import pandas as pd

from pywr.model import Model
from .custom_recorders import *
from bson.json_util import dumps
from .custom_parameters import *
from pywr.recorders.progress import ProgressRecorder
from pywr.recorders import TablesRecorder, CSVRecorder

# from wre_moea.libs.define_imports import * #cli_algorithm_search
# from wre_moea.libs.groups.algorithm_search.commands import cli_algorithm_search

import datetime

# from wre_moea.config import LOG_ALERT_PREFIX
# from wre_moea.libs.mixin.moea import (
#     InterfacePlatypusWrapper,
#     PyretoMongoPlatypusWrapper,
#     SaveNondominatedSolutionsArchive,
#     write_file_to_s3
# )



# Suppress warnings in current Pywr
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*Document requires version.*")  # optimisation/__init__.py:111
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Resampling with a PeriodIndex.*")  # dataframe_tools.py:127
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'M' is deprecated.*")  # dataframe_tools.py:127

import logging
logger = logging.getLogger(__name__)

def get_random_seed():
    import random
    return random.randint(2, 2**20-1)

@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    ch = logging.StreamHandler()

    loggers = [
        logger,
        logging.getLogger("pywr"),
        logging.getLogger("pywr_borg"),
    ]
    for _logger in loggers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        _logger.addHandler(ch)
        if debug:
            _logger.setLevel(logging.DEBUG)
        else:
            _logger.setLevel(logging.INFO)

    logger.info('Debug mode is %s' % ('on' if debug else 'off'))


@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def run(filename):
    """
    Run the Pywr model
    """

    logger.info('Loading model from file: "{}"'.format(filename))
    model = Model.load(filename, solver='glpk')

    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    
    ProgressRecorder(model)

    base, ext = os.path.splitext(filename)
    output_directory = os.path.join(base, "outputs")

    os.makedirs(os.path.join(output_directory), exist_ok=True)

    # TablesRecorder(model, os.path.join(output_directory, f"{base}_parameters.h5"), parameters=[p for p in model.parameters if p.name is not None])
    TablesRecorder(model, os.path.join(output_directory, f"{base}_parameters.h5"), parameters=[p for p in model.parameters if p.name is not None])
    CSVRecorder(model, os.path.join(output_directory, f"{base}_nodes.csv"))

    logger.info('Starting model run.')
    ret = model.run()
    logger.info(ret)
    print(ret.to_dataframe())

    # Save the recorders
    recorders_ = {}
    agg_recorders = {}

    for rec in model.recorders:
        
        try:
            recorders_[rec.name] = np.array(rec.values())
            agg_recorders[rec.name] = np.array(rec.aggregated_value())

        except NotImplementedError:
            pass

    recorders_ = pd.DataFrame(recorders_).T
    agg_recorders = pd.Series(agg_recorders, dtype=np.float64)

    writer = pd.ExcelWriter(os.path.join(output_directory, f"{base}_metrics.xlsx"))
    recorders_.to_excel(writer, 'values')
    agg_recorders.to_excel(writer, 'agg_values')

    if pd.__version__ >= '2.0.3':
        writer._save()
    else:
        writer.close()

    if any(s.size > 1 for s in model.scenarios.scenarios):
        # Save DataFrame recorders
        store = pd.HDFStore(os.path.join(output_directory, f"{base}_recorders.h5"), mode='w')

        for rec in model.recorders:
            if hasattr(rec, 'to_dataframe'):
                df = rec.to_dataframe()
                store[rec.name] = df

            try:
                values = np.array(rec.values())

            except NotImplementedError:
                pass
            
            else:
                store[f"{rec.name}_values"] = pd.Series(values)
        
        store.close()

    else:
        nmes = []
        rec_to_csv = []
        
        for rec in model.recorders:
            if hasattr(rec, 'to_dataframe'):
                df = rec.to_dataframe()
                nmes.append(rec.name)
                rec_to_csv.append(df) 

        rec_to_csv = pd.concat(rec_to_csv, axis=1)
        rec_to_csv.columns = nmes
        rec_to_csv.to_csv(os.path.join(output_directory, f"{base}_recorders.csv"))


@cli.command(name='run_simulation')
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def run_simulation(filename):
    """
    This method is used to run a pywr model and saving all the extra recorders/metrics we normally use in projects we have with The World Bank.
    """

    ''' -------- TEMPORAL FUNCTION -NEED TO FIX IT USING FILLFORWARD METHOD ON THE RECORDERS ----'''
    def convert_freq_to_days(freq):
        """
        Convert a frequency string to the number of days.
        """
        #print(f"VALUE freq: {freq}")
        if freq =='D':
            return 1
        elif freq.endswith('D'):
            return int(freq[:-1])
        elif freq == 'W':
            return 7
        elif freq == 'M':
            return 30.44  # Average number of days in a month
        elif freq == 'Y':
            return 365.25  # Average number of days in a year
        else:
            return freq
    ''' -------- ------------------------------------------------------------- ----'''

    logger.info('Loading model from file: "{}"'.format(filename))
    model = Model.load(filename, solver='glpk')

    # Silence Warnings
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*Resampling with a PeriodIndex is deprecated.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*'AS' is deprecated and will be removed in a future version, please use 'YS' instead.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.groupby with axis=1 is deprecated.*")
    
    ProgressRecorder(model)

    base, ext = os.path.splitext(filename)
    output_directory = os.path.join(base, "outputs")

    os.makedirs(os.path.join(output_directory), exist_ok=True)

    # Save DataFrame recorders
    store_metrics = pd.HDFStore(os.path.join(output_directory, f"{base}_metrics.h5"), mode='w')
    store_recorders = pd.HDFStore(os.path.join(output_directory, f"{base}_recorders.h5"), mode='w')
    store_aggreated = pd.HDFStore(os.path.join(output_directory, f"{base}_aggregated.h5"), mode='w')

    logger.info('Starting model run.')
    ret = model.run()
    logger.info(ret)
    print(ret.to_dataframe())

    for rec in model.recorders:
            
        if hasattr(rec, 'to_dataframe') and 'recorder' in rec.name:
            df = rec.to_dataframe()

            if model.timestepper.freq != df.index.freq:
                store_recorders[rec.name] = df

            else:
                if 'Hydropower Energy [MWh]' in rec.name:
                    store_recorders[rec.name] = df.resample('M').mean().multiply(30.42).loc[:str(model.timestepper.end.year),:] # Convert to MWh/month
                else:
                    store_recorders[rec.name] = df.resample('M').mean().loc[:str(model.timestepper.end.year),:]

        try:
            if 'Aggregated' in rec.name:
                values = np.array(rec.values())

        except NotImplementedError:
            pass
        
        else:
            if 'Aggregated' in rec.name:
                store_aggreated[rec.name] = pd.Series(values)
    
    store_recorders.close()
    store_aggreated.close()

    for rec in model.recorders:

        try:
            if ('Reliability' in rec.name or 'Resilience' in rec.name 
                or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                sc_index = model.scenarios.multiindex
                values = pd.DataFrame(np.array(rec.values()), index=sc_index)
                #values = np.array(rec.values()) #rec.values()

            if 'Hydropower Energy [MWh]' in rec.name:
                sc_index = model.scenarios.multiindex
                vals_hy = pd.DataFrame(np.array(rec.values()), index=sc_index)
                vals_hy.columns = [''] * len(vals_hy.columns)
                
                frq = convert_freq_to_days(model.timestepper.freq)  
                number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years
                factor_to_annual = frq / number_simulated_years  # 
                values_hydropower = vals_hy.multiply(factor_to_annual) # Convert to MWh/year     
                
                # methos 2 alternative (backup)
                # number_of_time_steps = (model.timestepper.end - model.timestepper.start).days/frq
                # values_hydropower = vals_hy.divide(number_of_time_steps).multiply(365) # Convert to MWh/year

            if 'Hydropower Firm Power [MW]' in rec.name:
                sc_index = model.scenarios.multiindex
                vals_pw = pd.DataFrame(np.array(rec.values()), index=sc_index)

                vals_pw.columns = [''] * len(vals_pw.columns)
                valures_firm_power = vals_pw

            if 'Volumetric Supply' in rec.name:
                sc_index = model.scenarios.multiindex

                vals_volumetric = pd.DataFrame(np.array(rec.values()), index=sc_index)

                number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years

                vals_volumetric.columns = [''] * len(vals_volumetric.columns)
                values_volumetric = vals_volumetric.divide(number_simulated_years) # Convert to Mm3/year

        except NotImplementedError:
            pass
        
        else:
            if ('Reliability' in rec.name or 'Resilience' in rec.name 
                or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                try:
                    store_metrics[f"{rec.name}"] = values
                except Exception as excp:
                    logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

            if 'Hydropower Energy [MWh]' in rec.name:
                try:
                    store_metrics[f"{rec.name}"] = values_hydropower
                except Exception as excp:
                    logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")
            
            if 'Hydropower Firm Power [MW]' in rec.name:
                try:
                    store_metrics[f"{rec.name}"] = valures_firm_power
                except Exception as excp:
                    logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

            if 'Volumetric Supply' in rec.name:
                try:
                    store_metrics[f"{rec.name}"] = values_volumetric
                except Exception as excp:
                    logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

    store_metrics.close()


@cli.command(name='run_dams_value')
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def run_dams_value(filename):
    """
    This method is used to run a pywr model for the AmuDary project with the WB
    """

    ''' -------- TEMPORAL FUNCTION -NEED TO FIX IT USING FILLFORWARD METHOD ON THE RECORDERS ----'''
    def convert_freq_to_days(freq):
        """
        Convert a frequency string to the number of days.
        """

        if freq =='D':
            return 1
        elif freq.endswith('D'):
            return int(freq[:-1])
        elif freq == 'W':
            return 7
        elif freq == 'M':
            return 30.44  # Average number of days in a month
        elif freq == 'Y':
            return 365.25  # Average number of days in a year
        else:
            return freq
    ''' -------- ------------------------------------------------------------- ----'''

    logger.info('Loading model from file: "{}"'.format(filename))

    dams_to_modify = {
        'All_dams': ['All_dams'],
        # 'Akdarya Dam': ['Akdarya Dam'],
        # 'Aktepin Dam': ['Aktepin Dam'],
        # 'Baipaza Dam': ['Baipaza Dam', 'Baipaza Dam Turbines'],# 'Baipaza Dam Control Release'],
        # 'Chimkurgan Dam': ['Chimkurgan Dam'],
        # 'Gissarak Dam': ['Gissarak Dam', 'Gissarak Dam Turbines'],# 'Gissarak Dam Control Release'],
        # 'Golovnaya Dam': ['Golovnaya Dam', 'Golovnaya Turbine'],# 'Golovnaya Control Release'],
        # 'Kamashi Dam': ['Kamashi Dam'],
        # 'Karasuv Dam': ['Karasuv Dam'],
        # 'Kattakurgan Dam': ['Kattakurgan Dam'],
        # 'Kumkurgan Dam': ['Kumkurgan Dam', 'Kumkurgan Dam Turbines'],# 'Kumkurgan Dam Control Release'],
        # 'Kuyumazar Dam': ['Kuyumazar Dam'],
        # 'Muminabad Dam': ['Muminabad Dam'],
        'Nurek Dam': ['Nurek Dam', 'Nurek Dam Turbines'],# 'Nurek Dam Control Release'],
        # 'Pachkamar Dam': ['Pachkamar Dam'],
        'Rogun Dam': ['Rogun Dam', 'Rogun Dam Turbines'], #'Rogun Dam Control Release'],
        # 'Sangtuda 1 Dam': ['Sangtuda 1 Dam', 'Sangtuda 1 Turbine'],# 'Sangtuda 1 Control Release'],
        # 'Sangtuda 2 Dam': ['Sangtuda 2 Dam', 'Sangtuda 2 Turbine'],# 'Sangtuda 2 Control Release'],
        # 'Talimardzhan Dam': ['Talimardzhan Dam'],
        'THC': ['THC Complex Dam in-channel', 'THC Complex Dam off-channel', 'THC Turbine'], #'THC Complex Dam in-channel Control Release'],
        # 'Tudakul Dam': ['Tudakul Dam'],
        # 'Tupalang Dam': ['Tupalang Dam', 'Tupalang Dam Turbines'], #'Tupalang Dam Control Release'],
        # 'Tusunsoy Dam': ['Tusunsoy Dam'],
        # 'Uchkyzyl Dam': ['Uchkyzyl Dam'],
        'Nurek and Rogun Dams': ['Rogun Dam', 'Rogun Dam Turbines', 'Nurek Dam', 'Nurek Dam Turbines']
        }

    for dam, items in dams_to_modify.items():

        with open(filename) as jfile:
            dta = json.load(jfile)

            for node in dta['nodes']:
                name = node.get('name')

                if name in items:
                    if 'Dam' in name and not 'Turbine' in name and not 'Control' in name:
                        node['min_volume'] = 0
                        node['max_volume'] = 0
                        node['initial_volume_pc'] = 0
                        #del node['level']
                        #del node['area']
                    # elif 'Control Release' in name:
                    #     node['max_flow'] = 100000
                    else:
                        node['max_flow'] = 0

        base, ext = os.path.splitext(filename)
        output_directory = os.path.join(f'{base}_outputs')

        os.makedirs(os.path.join(output_directory), exist_ok=True)

        with open(f'{base}_{dam}{ext}', 'w') as to_save:
            json.dump(dta, to_save, indent=4)

        model = Model.load(dta, solver='glpk')

        # Silence Warnings
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Resampling with a PeriodIndex is deprecated.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*'AS' is deprecated and will be removed in a future version, please use 'YS' instead.*")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.groupby with axis=1 is deprecated.*")
        
        ProgressRecorder(model)

        # Save DataFrame recorders
        store_metrics = pd.HDFStore(os.path.join(output_directory, f"{base}_{dam}_metrics.h5"), mode='w')
        store_recorders = pd.HDFStore(os.path.join(output_directory, f"{base}_{dam}_recorders.h5"), mode='w')
        store_aggreated = pd.HDFStore(os.path.join(output_directory, f"{base}_{dam}_aggregated.h5"), mode='w')

        logger.info('Starting model run.')
        ret = model.run()
        logger.info(ret)
        print(ret.to_dataframe())

        for rec in model.recorders:
                
            if hasattr(rec, 'to_dataframe') and 'recorder' in rec.name:
                df = rec.to_dataframe()

                if model.timestepper.freq != df.index.freq:
                    store_recorders[rec.name] = df

                else:
                    if 'Hydropower Energy [MWh]' in rec.name:
                        store_recorders[rec.name] = df.resample('M').mean().multiply(30.42).loc[:str(model.timestepper.end.year),:] # Convert to MWh/month
                    else:
                        store_recorders[rec.name] = df.resample('M').mean().loc[:str(model.timestepper.end.year),:]

            try:
                if 'Aggregated' in rec.name:
                    values = np.array(rec.values())

            except NotImplementedError:
                pass
            
            else:
                if 'Aggregated' in rec.name:
                    store_aggreated[rec.name] = pd.Series(values)
        
        store_recorders.close()
        store_aggreated.close()

        for rec in model.recorders:

            try:
                if ('Reliability' in rec.name or 'Resilience' in rec.name 
                    or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                    values = rec.values()

                if 'Hydropower Energy [MWh]' in rec.name:
                    sc_index = model.scenarios.multiindex
                    vals_hy = pd.DataFrame(np.array(rec.values()), index=sc_index)
                    vals_hy.columns = [''] * len(vals_hy.columns)
                    
                    frq = convert_freq_to_days(model.timestepper.freq)  
                    number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years
                    factor_to_annual = frq / number_simulated_years  # 
                    values_hydropower = vals_hy.multiply(factor_to_annual) # Convert to MWh/year     
                    
                    # methos 2 alternative (backup)
                    # number_of_time_steps = (model.timestepper.end - model.timestepper.start).days/frq
                    # values_hydropower = vals_hy.divide(number_of_time_steps).multiply(365) # Convert to MWh/year

                if 'Hydropower Firm Power [MW]' in rec.name:
                    sc_index = model.scenarios.multiindex
                    vals_pw = pd.DataFrame(np.array(rec.values()), index=sc_index)

                    vals_pw.columns = [''] * len(vals_pw.columns)
                    valures_firm_power = vals_pw

                if 'Volumetric Supply' in rec.name:
                    sc_index = model.scenarios.multiindex

                    vals_volumetric = pd.DataFrame(np.array(rec.values()), index=sc_index)

                    number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years

                    vals_volumetric.columns = [''] * len(vals_volumetric.columns)
                    values_volumetric = vals_volumetric.divide(number_simulated_years) # Convert to Mm3/year

            except NotImplementedError:
                pass
            
            else:
                if ('Reliability' in rec.name or 'Resilience' in rec.name 
                    or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                    try:
                        store_metrics[f"{rec.name}"] = values
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

                if 'Hydropower Energy [MWh]' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = values_hydropower
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")
                
                if 'Hydropower Firm Power [MW]' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = valures_firm_power
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

                if 'Volumetric Supply' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = values_volumetric
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

        store_metrics.close()


@cli.command(name='pyborg')
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-s', '--seed', type=int, default=None)
@click.option('-u', '--use-mpi', default=False)
@click.option('-n', '--max-nfe', type=int, default=1000)
@click.option('-f', '--frequency', type=int, default=None)
@click.option('-i', '--islands', type=int, default=1)
def pyborg(filename, seed, use_mpi, max_nfe, frequency, islands):

    from run_moea.borg import Configuration
    from run_moea.BsonBorgWrapper import PyretoJSONBorgWrapper
    
    directory, model_name = os.path.split(filename)
    output_directory = os.path.join(directory, 'outputs')

    if seed is None:
        seed = get_random_seed()
        random.seed(seed)

    if seed is not None:
        random.seed(seed)

    search_data = {'algorithm': 'Borg', 'seed': seed}

    runtime_file = f'{model_name[0:-5]}_seed-{seed}_runtime%d.txt'
    runtime_file_path = os.path.join(output_directory, runtime_file)

    wrapper = PyretoJSONBorgWrapper(filename, search_data=search_data, output_directory=output_directory, 
                                    model_name=model_name[0:-5], seed=seed)

    logger.info('Starting model search.')

    if use_mpi:
        Configuration.startMPI()
        results = wrapper.problem.solveMPI(islands=islands, maxEvaluations=max_nfe, runtime=runtime_file_path) #frequency=frequency,
        
        if results is not None:
            print(results)
        
        logger.info('Optimisation complete')

        Configuration.stopMPI()
    
    else:
        print(f"Running Borg with {max_nfe} evaluations")
        results = wrapper.problem.solve({"maxEvaluations": max_nfe, "runtimeformat": 'borg', "frequency": frequency,
                                "runtimefile": runtime_file_path})
        
        logger.info('Optimisation complete')

@cli.command(name='run_scenarios')
@click.option('-s', '--start', type=int, default=None)
@click.option('-e', '--end', type=int, default=None)
@click.option('-r', '--resample', default=False)
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def run_scenarios(filename, start, end, resample):
    """
    Run the Pywr model
    """

    ''' -------- TEMPORAL FUNCTION -NEED TO FIX IT USING FILLFORWARD METHOD ON THE RECORDERS ----'''
    def convert_freq_to_days(freq):
        """
        Convert a frequency string to the number of days.
        """
        print(f"VALUE freq: {freq}")
        if freq =='D':
            return 1
        elif freq.endswith('D'):
            return int(freq[:-1])
        elif freq == 'W':
            return 7
        elif freq == 'M':
            return 30.44  # Average number of days in a month
        elif freq == 'Y':
            return 365.25  # Average number of days in a year
        else:
            return freq
    ''' -------- ------------------------------------------------------------- ----'''

    logger.info('Loading model from file: "{}"'.format(filename))

    for i in range(start, end):

        with open(filename) as jfile:
            dta = json.load(jfile)

            dta['scenarios'][0]['slice'][0] = i
            dta['scenarios'][0]['slice'][1] = i+1
        
        model = Model.load(dta, solver='glpk')

        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
        
        ProgressRecorder(model)

        base, ext = os.path.splitext(filename)

        output_directory = os.path.join(base, f"outputs_{i}_{i+1}")

        os.makedirs(os.path.join(output_directory), exist_ok=True)

        # Save DataFrame recorders
        store_metrics = pd.HDFStore(os.path.join(output_directory, f"{base}_metrics.h5"), mode='w')
        store_recorders = pd.HDFStore(os.path.join(output_directory, f"{base}_recorders.h5"), mode='w')
        store_aggreated = pd.HDFStore(os.path.join(output_directory, f"{base}_aggregated.h5"), mode='w')

        logger.info('Starting model run.')
        ret = model.run()
        logger.info(ret)
        print(ret.to_dataframe())

        for rec in model.recorders:
                
            if hasattr(rec, 'to_dataframe') and 'recorder' in rec.name:
                df = rec.to_dataframe()

                if model.timestepper.freq != df.index.freq:
                    store_recorders[rec.name] = df

                else:
                    if 'Hydropower Energy [MWh]' in rec.name:
                        store_recorders[rec.name] = df.resample('M').mean().multiply(30.42).loc[:str(model.timestepper.end.year),:] # Convert to MWh/month
                    else:
                        store_recorders[rec.name] = df.resample('M').mean().loc[:str(model.timestepper.end.year),:]

            try:
                if 'Aggregated' in rec.name:
                    values = np.array(rec.values())

            except NotImplementedError:
                pass
            
            else:
                if 'Aggregated' in rec.name:
                    store_aggreated[rec.name] = pd.Series(values)
        
        store_recorders.close()
        store_aggreated.close()

        for rec in model.recorders:

            try:
                if ('Reliability' in rec.name or 'Resilience' in rec.name 
                    or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                    values = rec.values()

                if 'Hydropower Energy [MWh]' in rec.name:
                    sc_index = model.scenarios.multiindex
                    vals_hy = pd.DataFrame(np.array(rec.values()), index=sc_index)
                    vals_hy.columns = [''] * len(vals_hy.columns)
                    
                    frq = convert_freq_to_days(model.timestepper.freq)  
                    number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years
                    factor_to_annual = frq / number_simulated_years  # 
                    values_hydropower = vals_hy.multiply(factor_to_annual) # Convert to MWh/year     
                    
                    # methos 2 alternative (backup)
                    # number_of_time_steps = (model.timestepper.end - model.timestepper.start).days/frq
                    # values_hydropower = vals_hy.divide(number_of_time_steps).multiply(365) # Convert to MWh/year

                if 'Hydropower Firm Power [MW]' in rec.name:
                    sc_index = model.scenarios.multiindex
                    vals_pw = pd.DataFrame(np.array(rec.values()), index=sc_index)

                    vals_pw.columns = [''] * len(vals_pw.columns)
                    valures_firm_power = vals_pw

                if 'Volumetric Supply' in rec.name:
                    sc_index = model.scenarios.multiindex

                    vals_volumetric = pd.DataFrame(np.array(rec.values()), index=sc_index)

                    number_simulated_years = (model.timestepper.end - model.timestepper.start).days / 365.25   # total days of simulation div 365 to get total years

                    vals_volumetric.columns = [''] * len(vals_volumetric.columns)
                    values_volumetric = vals_volumetric.divide(number_simulated_years) # Convert to Mm3/year

            except NotImplementedError:
                pass
            
            else:
                if ('Reliability' in rec.name or 'Resilience' in rec.name 
                    or 'Annual Deficit' in rec.name or 'annual crop yield' in rec.name or 'supply reliability' in rec.name):
                    try:
                        store_metrics[f"{rec.name}"] = values
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

                if 'Hydropower Energy [MWh]' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = values_hydropower
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")
                
                if 'Hydropower Firm Power [MW]' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = valures_firm_power
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

                if 'Volumetric Supply' in rec.name:
                    try:
                        store_metrics[f"{rec.name}"] = values_volumetric
                    except Exception as excp:
                        logger.error(f"Error in saving data  in store_metrics:\n rec.name: {rec.name}.")

        store_metrics.close()


    # logger.info('Starting model run.')
    # ret = model.run()
    # logger.info(ret)
    # print(ret.to_dataframe())

    # Save DataFrame recorders
    # store = pd.HDFStore(os.path.join(output_directory, f"{base}_recorders.h5"), mode='w')

    # for rec in model.recorders:
        
    #     if hasattr(rec, 'to_dataframe'):
    #         df = rec.to_dataframe()

    #         if resample == True:
    #             df = df.resample('M').mean()
    #             store[rec.name] = df

    #         else:
    #             store[rec.name] = df

    #     try:
    #         values = np.array(rec.values())

    #     except NotImplementedError:
    #         pass
        
    #     else:
    #         store[f"{rec.name}_values"] = pd.Series(values)
    
    # store.close()


@cli.command(name='pywr_mpi_borg')
@click.option('-s', '--seed', type=int, default=None)
@click.argument('config_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def pywr_mpi_borg(config_file, seed):

    from pywr_borg import BorgMSModel, BorgRandomSeed
    from pywr_borg.core import logger as borg_logger
    from mpi4py import MPI

    with open(config_file) as config:
        dta = json.load(config)

    max_nfe = dta["search_configuration"]["max-nfe"]
    cluster = dta["search_configuration"]["cluster"]
    max_hours = dta["search_configuration"]["max_hours"]
    model_name = dta["search_configuration"]["model_name"]
    model_file = dta["search_configuration"]["model_file"]
    results_file = dta["search_configuration"]["results_file"]
    output_frequency = dta["search_configuration"]["output_frequency"]
    output_directory = dta["search_configuration"]["output_directory"]
    initial_population_archive = dta["search_configuration"]["initial_population_archive"]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if cluster == "UoM":
        BorgRandomSeed(seed)
    
    else:
        seed = dta["search_configuration"]["seed"]
        BorgRandomSeed(seed)

    if seed is None:
        seed = random.randrange(sys.maxsize)
        BorgRandomSeed(seed)

    out_dir = os.path.join(os.getcwd(), f'{output_directory}_{seed}')
    os.makedirs(out_dir, exist_ok=True)

    borg_logger.setLevel(logging.DEBUG)

    model = BorgMSModel.load(model_file)
    if rank == 0:
        logger.info("Setting up model")

        initial_population_archives = []
        if initial_population_archive is not None:
            initial_population_archives.append(initial_population_archive)

        model.setup(f'{model_name}_archive.json', initial_population_archives=initial_population_archives)

    else:
        model.setup()
    if rank == 0:
        logger.info("Running model")
    model.run(max_hours=max_hours, max_evaluations=max_nfe, output_runtime=f'{model_name}_runtime.txt',
              output_frequency=output_frequency)

    if rank == 0:
        logger.info('Optimisation complete')
        logger.info('{:d} solutions found in the Pareto-approximate set.'.format(len(model.archive)))
        with open(os.path.join(out_dir, results_file), mode='w') as fh:
            json.dump(model.archive_to_dict(), fh, sort_keys=True, indent=4)

        model.archive.printf()


@cli.command()
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('--use-mpi/--no-use-mpi', default=False)
@click.option('-s', '--seed', type=int, default=None)
@click.option('-p', '--num-cpus', type=int, default=None)
@click.option('-n', '--max-nfe', type=int, default=1000)
@click.option('--pop-size', type=int, default=50)
@click.option('-a', '--algorithm', type=click.Choice(['NSGAII', 'NSGAIII', 'EpsMOEA', 'EpsNSGAII']), default='NSGAII')
@click.option('-w', '--wrapper-type', type=click.Choice(['json', 'mongo', 'wpywr']), default='json')
@click.option('-e', '--epsilons', multiple=True, type=float, default=(0.05, ))
@click.option('--divisions-outer', type=int, default=12)
@click.option('--divisions-inner', type=int, default=0)
def search(filename, use_mpi, seed, num_cpus, max_nfe, pop_size, algorithm, wrapper_type, epsilons, divisions_outer, divisions_inner):
    import platypus
    from run_moea.BsonPlatypusWrapper import LoggingArchive , PyretoJSONPlatypusWrapper

    logger.info('Loading model from file: "{}"'.format(filename))
    directory, model_name = os.path.split(filename)
    output_directory = os.path.join(directory, 'outputs', f'{model_name[0:-5]}_{seed}')

    if algorithm == 'NSGAII':
        algorithm_klass = platypus.NSGAII
        algorithm_kwargs = {'population_size': pop_size}
    elif algorithm == 'NSGAIII':
        algorithm_klass = platypus.NSGAIII
        algorithm_kwargs = {'divisions_outer': divisions_outer, 'divisions_inner': divisions_inner}
    elif algorithm == 'EpsMOEA':
        algorithm_klass = platypus.EpsMOEA
        algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
    elif algorithm == 'EpsNSGAII':
        algorithm_klass = platypus.EpsMOEA
        algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
    else:
        raise RuntimeError('Algorithm "{}" not supported.'.format(algorithm))

    if seed is None:
        seed = random.randrange(sys.maxsize)

    search_data = {'algorithm': algorithm, 'seed': seed, 'user_metadata':algorithm_kwargs}
    if wrapper_type == 'json':
        wrapper = PyretoJSONPlatypusWrapper(filename, search_data=search_data, output_directory=output_directory)
    else:
        raise ValueError(f'Wrapper type "{wrapper_type}" not supported.')

    if seed is not None:
        random.seed(seed)

    logger.info('Starting model search.')

    # Use only to multi-node
    if use_mpi:

        from platypus.mpipool import MPIPool

        pool = MPIPool()
        evaluator_klass = platypus.PoolEvaluator
        evaluator_args = (pool,)

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    elif num_cpus is None:
        evaluator_klass = platypus.MapEvaluator
        evaluator_args = ()

    else:
        evaluator_klass = platypus.ProcessPoolEvaluator
        evaluator_args = (num_cpus,)

    with evaluator_klass(*evaluator_args) as evaluator:
        algorithm = algorithm_klass(wrapper.problem, evaluator=evaluator, **algorithm_kwargs, seed=seed)

        if wrapper_type == 'wpywr':
            algorithm.run(max_nfe, callback=wrapper.save_nondominant)
        else:
            algorithm.run(max_nfe)

    # Use only to multi-node
    if use_mpi:
        pool.close()



# @cli_algorithm_search.command("platypus")
# @click.pass_context
# @click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
# @click.option('--output-dir', type=click.Path(file_okay=False, dir_okay=True, exists=True))
# @click.option('--use-mpi/--no-use-mpi', default=False)
# @click.option('-s', '--seed', type=int, default=None)
# @click.option('-p', '--num-cpus', type=int, default=None)
# @click.option('-n', '--max-nfe', type=int, default=1000)
# @click.option('--pop-size', type=int, default=50)
# @click.option('-a', '--algorithm', type=click.Choice(['NSGAII', 'NSGAIII', 'EpsMOEA', 'EpsNSGAII']), default='NSGAII')
# @click.option('-w', '--wrapper-type', type=click.Choice(['json', 'mongo', 'wpywr']), default='json')
# @click.option('-e', '--epsilons', multiple=True, type=float, default=(0.05, ))
# @click.option('--divisions-outer', type=int, default=12)
# @click.option('--divisions-inner', type=int, default=0)
# @click.option('--s3-output', is_flag=True, default=False)
# @click.option('--polyvis', is_flag=True, default=False)
# def platypus_search_command(ctx, filename, output_dir, use_mpi, seed, num_cpus, max_nfe, pop_size, algorithm, wrapper_type, epsilons, divisions_outer, divisions_inner, s3_output, polyvis):
#     """Run a search using Platypus"""

#     return platypus_search(filename, s3_output, output_dir, use_mpi, seed, num_cpus, max_nfe, pop_size, algorithm,
#                            wrapper_type, epsilons, divisions_outer, divisions_inner, polyvis)

# def platypus_search(filename, s3_output, output_dir="/tmp", use_mpi=False, seed=None, num_cpus=None,
#                     max_nfe=1000, pop_size=50, algorithm='NSGAIII', wrapper_type='json',
#                     epsilons=(0.05, ), divisions_outer=12, divisions_inner=0, polyvis=False):
#     logger.info("Running platypus search")
#     import platypus

#     logger.info('Loading model from file: "{}"'.format(filename))
#     directory, model_name = os.path.split(filename)
#     output_directory = 'outputs'#:wos.path.join(directory, 'outputs')

#     if algorithm == 'NSGAII':
#         from platypus.algorithms import NSGAII
#         algorithm_klass = NSGAII
#         algorithm_kwargs = {'population_size': pop_size}
#     elif algorithm == 'NSGAIII':
#         from platypus.algorithms import NSGAIII
#         algorithm_klass = NSGAIII
#         algorithm_kwargs = {'divisions_outer': divisions_outer, 'divisions_inner': divisions_inner}
#     elif algorithm == 'EpsMOEA':
#         from platypus.algorithms import EpsMOEA
#         algorithm_klass = EpsMOEA
#         algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
#     elif algorithm == 'EpsNSGAII':
#         from platypus.algorithms import EpsNSGAII
#         algorithm_klass = EpsNSGAII
#         algorithm_kwargs = {'population_size': pop_size, 'epsilons': epsilons}
#     else:
#         raise RuntimeError('Algorithm "{}" not supported.'.format(algorithm))

#     if seed is None:
#         seed = random.randint(2, 2**32-1)

#     search_data = {'algorithm': algorithm, 'seed': seed, 'user_metadata': algorithm_kwargs}
#     if wrapper_type == 'json':
#         output_directory += '_json'
#         wrapper = InterfacePlatypusWrapper(
#         max_nfe,
#         filename,
#         use_mpi=use_mpi,
#         s3_output=s3_output,
#         polyvis=polyvis,
#         seed=seed,
#         num_cpus=num_cpus,
#         pop_size=pop_size,
#         algorithm=algorithm,
#         wrapper_type=wrapper_type,
#         epsilons=epsilons,
#         divisions_outer=divisions_outer,
#         divisions_inner=divisions_inner,
#         output_dir=output_dir)
#     elif wrapper_type == 'mongo':
#         wrapper = PyretoMongoPlatypusWrapper(filename, search_data=search_data, db='volta',
#                                              uri='mongodb://root:ahmoo8uti4Fa@34.240.214.74:27017/')
#     elif wrapper_type == 'wpywr':
#         output_directory += '_archive'
#         wrapper = SaveNondominatedSolutionsArchive(filename, search_data=search_data, output_directory=output_directory,
#                                                    model_name=model_name)
#     else:
#         raise ValueError(f'Wrapper type "{wrapper_type}" not supported.')

#     logger.info('Starting model search.')
#     wrapper.make_run_metadata(filename, algorithm, seed, max_nfe, pop_size, **algorithm_kwargs)

#     if use_mpi:
#         from wre_moea.libs.mixin.mpipool import HydraMPIPool

#         try:
#             pool = HydraMPIPool(debug=True)
#         except ValueError:
#             logger.critical("A Platypus MPIPool requires at least two processes")
#             exit(1)
#         evaluator_klass = platypus.PoolEvaluator
#         evaluator_args = (pool,)

#         if not pool.is_master():
#             pool.wait()
#             exit(0)

#     elif num_cpus is None:
#         from platypus import MapEvaluator
#         evaluator_klass = MapEvaluator
#         evaluator_args = ()

#     else:
#         from platypus import ProcessPoolEvaluator
#         evaluator_klass = ProcessPoolEvaluator
#         evaluator_args = (num_cpus,)

#     with evaluator_klass(*evaluator_args) as evaluator:
#         algorithm = algorithm_klass(wrapper.problem.problem, evaluator=evaluator, **algorithm_kwargs, seed=seed)

#         if wrapper_type == 'wpywr':
#             algorithm.run(max_nfe, callback=wrapper.save_nondominant)
#         else:
#             algorithm.run(max_nfe, callback=wrapper.metrics_callback)
#             wrapper.output["metadata"]["end"] = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
#             if output_dir:
#                 output_file = wrapper.write_output(output_dir)
#                 if s3_output:
#                     output_url = write_file_to_s3(output_file)
#                     print(f"{LOG_ALERT_PREFIX}{output_url}", flush=True)
#             if polyvis:
#                 from wre_moea.libs.consumers import PolyvisConsumer
#                 pc = PolyvisConsumer(wrapper.output)
#                 polyvis_url = pc.process()
#                 wrapper.output["metadata"]["polyvis"] = polyvis_url
#                 print(f"{LOG_ALERT_PREFIX}{polyvis_url}", flush=True)

#     # Use only to multi-node
#     if use_mpi:
#         pool.close()


#@cli.command(name='pywr_borg')
#@click.option('-s', '--seed', type=int, default=None)
#@click.argument('config_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
#def pywr_borg(config_file, seed):

#    from pywr_borg import BorgModel, BorgRandomSeed

#    with open(config_file) as config:
#        dta = json.load(config)

#    max_nfe = dta["search_configuration"]["max-nfe"]
#    cluster = dta["search_configuration"]["cluster"]
#    model_file = dta["search_configuration"]["model_file"]
#    results_file = dta["search_configuration"]["results_file"]
#    output_directory = dta["search_configuration"]["output_directory"]

#    if cluster == "UoM":
#        BorgRandomSeed(seed)
    
#    else:
#        seed = dta["search_configuration"]["seed"]
#        BorgRandomSeed(seed)

#    if seed is None:
#        seed = random.randrange(sys.maxsize)
#        BorgRandomSeed(seed)

#    out_dir = os.path.join(os.getcwd(), f'{output_directory}_{seed}')
#    os.makedirs(out_dir, exist_ok=True)

#    model = BorgModel.load(model_file)
    
#    logger.info("Setting up model")
#    model.setup()
    
#    logger.info("Running model")
#    model.run(max_nfe)
#    logger.info('Optimisation complete')
    
#    logger.info('{:d} solutions found in the Pareto-approximate set.'.format(len(model.archive)))
    
    # Log run statistics.
#    with open(os.path.join(out_dir, results_file), mode='w') as fh:

#        json.dump(model.archive_to_dict(), fh, sort_keys=True, indent=4)

#    model.archive.printf()


if __name__ == '__main__':
    cli()


def start_cli():
    cli()