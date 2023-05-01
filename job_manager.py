#!/usr/bin/env python

## dependencies
import os
import multiprocessing
import argparse
from joblib import Parallel, delayed

num_qubit_list = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]

## number of cores to use (if parallel)
MAX_CORES = 12
num_cores = min(MAX_CORES, multiprocessing.cpu_count() - 1) #leave 1 free

parser = argparse.ArgumentParser(
    description='Train multiple instances of the Clifford synthesizer.'
)

parser.add_argument(
    '-parallel',
    default=False,
    action=argparse.BooleanOptionalAction,
    help='boolean flag used to control parallel execution'
    )

args = vars(parser.parse_args())

## run in serial or parallel
if not args['parallel']:

    print('running in serial')

    for num_qubits in num_qubit_list:
        command_string = f'python clifford.py --num_qubits {num_qubits} --drop_phase_bits' # --no-load_saved_model'
        os.system(command_string)

else:
    print('running in parallel')
    print("number of cores we will use: ", num_cores)

    ## run the python scripts via command line
    def process_input(i):
        print("job number: ", i+1)
        num_qubits = num_qubit_list[i]
        command_string = f'python clifford.py -num_qubits {num_qubits}'
        return os.system(command_string)

    ## when there are less jobs then number of cores, do them all at once
    if len(num_qubit_list) <= num_cores:
        Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in range(len(num_qubit_list)))

    ## when there are more jobs then number of cores, do them in batches
    if len(num_qubit_list) > num_cores:
        n = len(num_qubit_list)//num_cores
        r = len(num_qubit_list) % num_cores

        for j in range(n):
            Parallel(n_jobs=num_cores)(delayed(process_input)(i) for i in range(j*num_cores, (j+1)*num_cores))

        # the remainder
        if r != 0:
            Parallel(n_jobs=r)(delayed(process_input)(i) for i in range(num_cores*n, num_cores*n + r))