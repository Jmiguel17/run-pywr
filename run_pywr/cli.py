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

import logging
logger = logging.getLogger(__name__)

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

    #TablesRecorder(model, os.path.join(output_directory, f"{base}_parameters.h5"), parameters=[p for p in model.parameters if p.name is not None])
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

@cli.command(name='borg_optimise')
@click.argument('filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-s', '--seed', type=int, default=None)
@click.option('-u', '--use-mpi', default=False)
@click.option('-n', '--max-nfe', type=int, default=1000)
@click.option('-f', '--frequency', type=int, default=500)
def borg_optimise(filename, seed, use_mpi, max_nfe, frequency):

    from run_moea.BorgWrapper import BorgWrapper

    wrapper = BorgWrapper(filename, seed, max_nfe, use_mpi, frequency)
    wrapper.run()


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
        seed = random.randrange(sys.maxsize)

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

    logger.info('Loading model from file: "{}"'.format(filename))

    with open(filename) as jfile:
        dta = json.load(jfile)

        dta['scenarios'][0]['slice'][0] = start
        dta['scenarios'][0]['slice'][1] = end
    
    model = Model.load(dta, solver='glpk')

    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    
    ProgressRecorder(model)

    base, ext = os.path.splitext(filename)

    output_directory = os.path.join(base, f"outputs_{start}_{end}")

    os.makedirs(os.path.join(output_directory), exist_ok=True)

    logger.info('Starting model run.')
    ret = model.run()
    logger.info(ret)
    print(ret.to_dataframe())

    # Save DataFrame recorders
    store = pd.HDFStore(os.path.join(output_directory, f"{base}_recorders.h5"), mode='w')

    for rec in model.recorders:
        
        if hasattr(rec, 'to_dataframe'):
            df = rec.to_dataframe()

            if resample == True:
                df = df.resample('M').mean()
                store[rec.name] = df

            else:
                store[rec.name] = df

        try:
            values = np.array(rec.values())

        except NotImplementedError:
            pass
        
        else:
            store[f"{rec.name}_values"] = pd.Series(values)
    
    store.close()


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
    output_directory = os.path.join(directory, 'outputs')

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