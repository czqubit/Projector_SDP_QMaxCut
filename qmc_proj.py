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

import itertools as it
import functools as ft
import numpy as np
import scipy as sp
from typing import List, Tuple
import time
import mosek
from   mosek.fusion import *
import sys
from pathlib import Path

def qmc_exact_weighted(n, edges):
    r"""
    Exact diagonalization for the quantum maxcut problem. 
    H = \sum_{(i,j)} w_{ij} * (I - X_iX_j - Y_iY_j - Z_iZ_j) / 4

    Return the maximum eigenvalue of H.

    Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].

    """

    I = sp.sparse.csr_matrix(np.eye(2).astype(np.complex128))
    X = sp.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex128))
    Y = sp.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex128))
    Z = sp.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex128))

    # sparse kron product
    def sparse_kron(A, B):
        return sp.sparse.kron(A, B, format='csr')
    
    # return op, e.g., (X,X), acting on target qubit, e.g., (0,1), tensored with I on other qubits
    def op_q(op, edge, n):
        tprod = [I for _ in range(n)]
        for i, j in enumerate(edge):
            tprod[j] = op[i]
        
        return ft.reduce(sparse_kron, tprod)

    # return density matrix for a singlet acting on qubits edge (i,j) tensored with I on other D-2 qubits
    def singlet(edge, n):
        return op_q((X, X), edge, n) + op_q((Y, Y), edge, n) + op_q((Z, Z), edge, n)
    
    H = 0
    weight_sum = 0
    for i,j,w in edges:
        H += w * singlet((i,j), n)
        weight_sum += w

    energy_min, v = sp.sparse.linalg.eigsh(H, k=1, which='SA')
    energy_max = (weight_sum - energy_min.item()) / 4
    
    return energy_max

def proj_mult(a: Tuple, b: Tuple) -> Tuple:
    r"""
    Define the multiplication algebra for a and b, 
    where a and b are tuples indexing the projectors, 
    e.g., a = (0,1) stands for projector h_{01}.

    The algebra: <,> is the expectation value
    .. math::
    <h_{00}> = 1
    <h_{ij}^2> = <h_{ij}>
    <h_{ij}h_{kl}> = <h_{kl}h_{ij}> if {i,j} \cap {k,l} = \0
    <h_{ij}h_{ik}> = 1/4 * (<h_ij> + <h_ik> - <h_jk>) if j \neq k
    The last equality applies to other sets that shares one common index.

    Args:
        *a (tuple)*: a tuple with indices of a projector h
        *b (tuple)*: a tuple with indices of a projector h

    Returns (tuple): (coefficient, product, flag)

    """
    # Check if a and b are tuples
    assert isinstance(a, tuple) and isinstance(b, tuple)

    # Take care special cases when a = (0,0) = I or b = (0,0) = I,
    # and a = (i,j) = b
    # flag denotes the number of shared indices between a and b,
    # flag = 2 if a = (0,0) or b = (0,0) or a = b = (0,0)
    if a == (0,0) and b != (0,0):
        coeff, prod, flag = (1,), (b,), 2
    elif a != (0,0) and b == (0,0):
        coeff, prod, flag = (1,), (a,), 2
    elif a == b:
        coeff, prod, flag = (1,), (a,), 2
    else:
        a_set = set(a)
        b_set = set(b)
        c = a_set.intersection(b_set)
        u = a_set.union(b_set)
        if len(c) == 0:
            coeff, prod, flag = (1,), ((a,b),), 0
        elif len(c) == 2:
            coeff, prod, flag = (1,), (a,), 2
        elif len(c) == 1:
            # get the single element from set c
            (e,) = c
            u.remove(e)
            coeff, prod, flag = (1/4, 1/4, -1/4), (a, b, tuple(sorted(tuple(u)))), 1

    return coeff, prod, flag
    
def get_data_main(n, edges):
    r"""
    Get all the data necessary for mosek fusion modeling in the primal formulation:
            max  sum_{(i,j)\in E} h_{ij}
            s.t. M(h_{ij}) >= 0,
    where M is the moment matrix, and '>=' stands for positive semidefiniteness. 

    Matrix indexing is column based. 

    Edges are not weighted, i.e., edges = [(i_1,j_1),...,(i_s,j_s)]. 
    
    """

    # Get all the h tuples (i,j), i < j and (0,0)
    htuples = [(0,0)] + list(it.combinations(range(n),2))
    n_htuples = len(htuples)
    # Dictionary storing the positions of the constraint variables, i.e., all the h_{ij}'s. 
    var_dict_constr_pos = {h: (i, 0) for i, h in enumerate(htuples)}
    # Dictionary storing the positions of the free variables, i.e., all the h_{ij}h_{kl}, 
    # where there's no overlap between {i,j} and {k, l}.
    var_dict_free_pos = {}
    # A stores all sparse matrices that specify the equality constraints in moment matrix M such that Tr(M A_i) = 0
    A = {}
    for i in range(n_htuples):
        for j in range(i, n_htuples):
            a, b = htuples[i], htuples[j]
            if i == 0 and j >= 1:
                A[(a,b)] = [[], [], []]
                A[(a,b)][0] = [j, j]
                A[(a,b)][1] = [i, j]
                A[(a,b)][2] = [1, -1]
            elif i > 0 and j > i: 
                a_set = set(a)
                b_set = set(b)
                c = a_set.intersection(b_set)
                u = a_set.union(b_set)
                if len(c) == 0:
                    var_dict_free_pos[(a,b)] = (j,i)
                elif len(c) == 1:
                    # get the single element from set c
                    (e,) = c
                    u.remove(e)
                    pos_a = var_dict_constr_pos[a]
                    pos_b = var_dict_constr_pos[b]
                    pos_c = var_dict_constr_pos[tuple(sorted(tuple(u)))]
                    A[(a,b)] = [[], [], []]
                    A[(a,b)][0] = [j, pos_a[0], pos_b[0], pos_c[0]]
                    A[(a,b)][1].extend([i, pos_a[1], pos_b[1], pos_c[1]])
                    A[(a,b)][2].extend([-1, 1/4, 1/4, -1/4])

    # Obtain matrix C in sparse format
    C = [[], [], []]
    for i, j in edges:
        pos = var_dict_constr_pos[(i,j)]
        C[0].append(pos[0])
        C[1].append(pos[1])
        C[2].append(1)

    # Dimension of moment matrix M
    X_dim = int(n * (n-1) / 2 + 1)

    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos

def get_data_weighted(n, edges):
    r"""
    Get all the data necessary for mosek fusion modeling in the primal formulation:
            max  sum_{(i,j)\in E} w_{ij} * h_{ij}
            s.t. M(h_{ij}) >= 0,
    where M is the moment matrix, and '>=' stands for positive semidefiniteness. 

    Matrix indexing is column based. 
    
    Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].

    """

    # Get all the h tuples (i,j), i < j and (0,0)
    htuples = [(0,0)] + list(it.combinations(range(n),2))
    n_htuples = len(htuples)
    # Dictionary storing the positions of the constraint variables, i.e., all the h_{ij}'s. 
    var_dict_constr_pos = {h: (i, 0) for i, h in enumerate(htuples)}
    # Dictionary storing the positions of the free variables, i.e., all the h_{ij}h_{kl}, 
    # where there's no overlap between {i,j} and {k, l}.
    var_dict_free_pos = {}
    # A stores all sparse matrices that specify the equality constraints in moment matrix M such that Tr(M A_i) = 0
    A = {}
    for i in range(n_htuples):
        for j in range(i, n_htuples):
            a, b = htuples[i], htuples[j]
            if i == 0 and j >= 1:
                A[(a,b)] = [[], [], []]
                A[(a,b)][0] = [j, j]
                A[(a,b)][1] = [i, j]
                A[(a,b)][2] = [1, -1]
            elif i > 0 and j > i: 
                a_set = set(a)
                b_set = set(b)
                c = a_set.intersection(b_set)
                u = a_set.union(b_set)
                if len(c) == 0:
                    var_dict_free_pos[(a,b)] = (j,i)
                elif len(c) == 1:
                    # get the single element from set c
                    (e,) = c
                    u.remove(e)
                    pos_a = var_dict_constr_pos[a]
                    pos_b = var_dict_constr_pos[b]
                    pos_c = var_dict_constr_pos[tuple(sorted(tuple(u)))]
                    A[(a,b)] = [[], [], []]
                    A[(a,b)][0] = [j, pos_a[0], pos_b[0], pos_c[0]]
                    A[(a,b)][1].extend([i, pos_a[1], pos_b[1], pos_c[1]])
                    A[(a,b)][2].extend([-1, 1/4, 1/4, -1/4])

    # Obtain matrix C in sparse format
    C = [[], [], []]
    for i, j, w in edges:
        pos = var_dict_constr_pos[(i,j)]
        C[0].append(pos[0])
        C[1].append(pos[1])
        C[2].append(w)

    # Dimension of moment matrix M
    X_dim = int(n * (n-1) / 2 + 1)

    return A, C, X_dim, var_dict_constr_pos, var_dict_free_pos

def solve_main_weighted(n, edges):
    """Main functioin for level-1 projector SDP for QMAXCUT. 

    Edges are weighted, i.e., edges = [(i_1, j_1, w_{i_1j_1}),...,(i_s, j_s, w_{i_sj_s})].
    
    """

    A, C, X_dim, var_dict_constr_pos, var_dict_free_pos = get_data_weighted(n, edges)

    with Model("qmc") as M:
        
        # Setting up the variables
        X = M.variable("X", Domain.inPSDCone(X_dim))

        # Objective
        C = Matrix.sparse(X_dim, X_dim, C[0], C[1], C[2])
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(C, X))

        # Constraints
        B = [ Matrix.sparse(X_dim, X_dim, a[0], a[1], a[2]) for a in A.values() ]
        M.constraint(X.index([0,0]), Domain.equalsTo(1.0))
        [ M.constraint(Expr.dot(b, X), Domain.equalsTo(0.0)) for b in B ]

        # Solve
        M.setLogHandler(sys.stdout)
        M.solve()

        # Get the objective value, moments, and the moment matrix (=S)
        obj_val = M.primalObjValue()
        X_sol = X.level()
        # moment matrix
        MM = np.reshape(X_sol, (X_dim,X_dim))
        # Correlations, first row of the moment matrix
        # correlations = {var:  MM[pos[0], pos[1]] for var, pos in var_dict_constr_pos.items()}
        # free moments, i.e., <h_ij h_kl> where no overlap between {i,j} and {k,l}
        # moments_free = {var: MM[pos[0], pos[1]] for var, pos in var_dict_free_pos.items()}
        # All the unique moments
        # moments = correlations | moments_free

    return obj_val, MM


if __name__=='__main__':


    # Example: ring-n with uniform weights 1.
    n = 4
    edges = [(i, i+1, 1) for i in range(n-1)] + [(0,n-1,1)]
    graph_name = f'ring-{n}'
    print(edges)

    dir_result = './ring/'
    # Make file path if not existed
    Path(dir_result).mkdir(parents=True, exist_ok=True)

    time_start = time.time()
    E_proj, moment_matrix = solve_main_weighted(n, edges)
    time_end_proj = time.time()
    t_proj = np.round(time_end_proj - time_start, 2)

    E_exact = qmc_exact_weighted(n, edges)
    time_end_exact = time.time()
    t_exact = np.round(time_end_exact - time_end_proj)
    E_diff = np.abs(E_exact - E_proj)

    print(f'The exact energy: \nE = {E_exact}')
    print(f'The approximated energy: \nL = {E_proj}')
    print(f'The difference: \nD = {E_diff}')
    print(f'Total time elapsed proj: {t_proj}')
    print(f'Total time elapsed exact: {t_exact}')
    # Save statistics
    with open(dir_result + f'{graph_name}_stats.txt', 'w') as f:
        f.write(f'{graph_name} {E_exact} {E_proj} {E_diff}\n')
    # Save data
    with open(dir_result + f'{graph_name}_mm.npy', 'wb') as f:
        np.save(f, moment_matrix)
