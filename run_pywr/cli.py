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

    TablesRecorder(model, os.path.join(output_directory, f"{base}_Parameters.h5"), parameters=[p for p in model.parameters if p.name is not None])
    CSVRecorder(model, os.path.join(output_directory, f"{base}_Nodes.csv"))

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

    writer = pd.ExcelWriter(os.path.join(output_directory, f"{base}_Metrics.xlsx"))
    recorders_.to_excel(writer, 'values')
    agg_recorders.to_excel(writer, 'agg_values')
    writer.save()

    # Save DataFrame recorders
    store = pd.HDFStore(os.path.join(output_directory, f"{base}_Recorders.h5"), mode='w')

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


@cli.command(name='pyborg')
@click.argument('config_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-s', '--seed', type=int, default=None)
def pyborg(config_file, seed):
    
    from run_moea.borg import Configuration
    from run_moea.BsonBorgWrapper import LoggingArchive , PyretoJSONBorgWrapper

    with open(config_file) as config:
        dta = json.load(config)

    max_nfe = dta["search_configuration"]["max-nfe"]
    use_mpi = dta["search_configuration"]["use-mpi"]
    model_file = dta["search_configuration"]["model_file"]

    logger.info('Loading model from file: "{}"'.format(model_file))

    out_dir = dta["search_configuration"]["output_directory"]
    model_name = dta["search_configuration"]["model_name"]

    cluster = dta["search_configuration"]["cluster"]

    if cluster == "UoM":
        search_data = {'algorithm': 'Borg', 'seed': seed}
    
    else:
        seed = dta["search_configuration"]["seed"]
        search_data = {'algorithm': 'Borg', 'seed': seed}

    if seed is None:
        seed = random.randrange(sys.maxsize)

    wrapper = PyretoJSONBorgWrapper(model_file, search_data=search_data, output_directory=out_dir, model_name=model_name, seed=seed)

    if seed is not None:
        random.seed(seed)

    logger.info('Starting model search.')

    if use_mpi:
        Configuration.startMPI()
        wrapper.problem.solveMPI(islands=1, maxEvaluations=max_nfe)

    else:
        wrapper.problem.solve({"maxEvaluations": max_nfe})

    if use_mpi:
        Configuration.stopMPI()


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


@cli.command(name='pywr_borg')
@click.option('-s', '--seed', type=int, default=None)
@click.argument('config_file', type=click.Path(file_okay=True, dir_okay=False, exists=True))
def pywr_borg(config_file, seed):

    from pywr_borg import BorgModel, BorgRandomSeed

    with open(config_file) as config:
        dta = json.load(config)

    max_nfe = dta["search_configuration"]["max-nfe"]
    cluster = dta["search_configuration"]["cluster"]
    model_file = dta["search_configuration"]["model_file"]
    results_file = dta["search_configuration"]["results_file"]
    output_directory = dta["search_configuration"]["output_directory"]

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

    model = BorgModel.load(model_file)
    
    logger.info("Setting up model")
    model.setup()
    
    logger.info("Running model")
    model.run(max_nfe)
    logger.info('Optimisation complete')
    
    logger.info('{:d} solutions found in the Pareto-approximate set.'.format(len(model.archive)))
    
    # Log run statistics.
    with open(os.path.join(out_dir, results_file), mode='w') as fh:

        json.dump(model.archive_to_dict(), fh, sort_keys=True, indent=4)

    model.archive.printf()


if __name__ == '__main__':

    cli()


def start_cli():

    cli()