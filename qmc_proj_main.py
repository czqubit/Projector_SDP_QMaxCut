# Copyright 2023 Cunlu Zhou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Example script showing how to run qmc_proj tasks in batches. 
# The two models we use here are the Majumdar-Ghosh model and 
# the Shastry-Sutherland model with different sizes of N. 

from qmc_proj import qmc_exact_weighted, solve_main_weighted
import numpy as np
import time
import argparse
from pathlib import Path


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Set parameters')
    parser.add_argument('--arrayID', default=None)
    parser.add_argument('--graph', default=None)

    args = parser.parse_args()
    arrayID = int(args.arrayID) 
    graph = str(args.graph)

    assert graph == 'MG' or graph == 'SS', \
    f"Graph must be either 'MG' (Majumdar-Ghosh) or 'SS' (Shastry-Sutherland); {graph} is given." 

    # There are two types of weights: when edge[2] == 1, w = 1; 
    # when edge[2] == 2, w = x. For Majumdar-Ghosh, x \in [0, 2], 
    # and for Shastry-Sutherland, x \in [1/2, 3]. We are using a mesh for x with 1/40 = 0.025.
    # So for Majumdar-Ghosh, arrayID=0-80, and x = arrayID * 0.025; for Shastry-Sutherland, 
    # arrayID=0-100, and x = 0.5 + arrayID * 0.025. 

    # extract graph
    n = 64
    if graph == 'MG':
        graph_name = f'Majumdar-Ghosh_N{n}'
        x = arrayID * 0.025
    elif graph == 'SS':
        graph_name = f'Shastry-Sutherland_N{n}'
        x = 0.5 + arrayID * 0.025
    graph_path = f'./graphdata/{graph_name}.dat'
    edges = np.loadtxt(graph_path, dtype=int).tolist()
    # Set the correct weight for type-2 edges.
    for e in edges:
        if e[2] == 2:
            e[2] = x
    edges = [tuple(sorted(e[0:2]) + [e[2]]) for e in edges]

    dir_result = f'./{graph_name}/'
    # Create the result directory if not existed
    Path(dir_result).mkdir(parents=True, exist_ok=True)

    print(f'arrayID, graph = {arrayID, graph_name}\n')
    print(f'edges = {edges}\n')

    time_start = time.time()
    E_proj, moment_matrix, correlations, moments_free, moments = solve_main_weighted(n, edges)
    time_end_lasserre = time.time()
    t_proj = np.round(time_end_lasserre - time_start, 2)
    # E_exact = qmc_exact_weighted(n, edges)
    E_exact = 'NA'
    # time_end_exact = time.time()
    # t_exact = np.round(time_end_exact - time_end_lasserre)
    # E_diff = np.abs(E_exact - E_proj)
    E_diff = 'NA'
    # print(f'The exact energy: \nE = {E_exact}')
    print(f'The approximated energy: \nL = {E_proj}')
    print(f'The difference: \nD = {E_diff}')
    print(f'Total time elapsed proj: {t_proj}')
    # print(f'Total time elapsed exact: {t_exact}')
    # Save statistics
    with open(dir_result + f'{graph_name}_stats_x_{x:.3f}.txt', 'w') as f:
        f.write(f'{graph_name} {E_exact} {E_proj} {E_diff}\n')
    # Save data
    with open(dir_result + f'{graph_name}_data_x_{x:.3f}.npy', 'wb') as f:
        np.save(f, moment_matrix)