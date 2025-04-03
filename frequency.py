# Copyright 2021 D-Wave Systems Inc.
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

import numpy as np

# Ignore errors importing matplotlib.pyplot (may not be available in testing framework)
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

import dimod
from qdeepsdk import QDeepHybridSolver

from philadelphia import load_problem, get_forbidden_set, plot_nodes
from utilities import check_results, get_frequencies, print_frequency_separations

def construct_bqm(demand, nfreq, reuse_distances, penalty_coef=1.0):
    """Construct BQM for feasibility frequency assignment problem.
    
    Args:
        demand (dict):
            Dictionary mapping each node number to a demand value
        nfreq (int):
            Number of frequencies to consider
        reuse_distances (list):
            List of reuse distances
        penalty_coef (float):
            Penalty coefficient associated with constraint penalty
            function.  Not needed in current formulation, which does
            not include an objective component of the problem formulation.
            Retained only as a placeholder in case the problem is extended
            to include an objective.

    Returns:
        dimod.BinaryQuadraticModel
    """
    # Variables:
    # x_vf, for v in nodes and f in frequencies: Is frequency f assigned to node v?
    nodes = sorted(list(demand.keys()))
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Constraints to enforce demand at each node:
    # Sum_f[ (1-2C) x_vf ] + Sum_j>i[ 2 x_vfi x_vfj ] + C^2
    # Linear parts:
    for v in nodes:
        for f in range(nfreq):
            var = f'x_{v}_{f}'
            bqm.add_variable(var, penalty_coef * (1.0 - 2 * demand[v]))
    # Interactions to enforce that each node gets the correct number of frequencies:
    for v in nodes:
        for fi in range(nfreq):
            for fj in range(fi + 1, nfreq):
                var_i = f'x_{v}_{fi}'
                var_j = f'x_{v}_{fj}'
                bqm.add_interaction(var_i, var_j, penalty_coef * 2.0)

    # Define penalties associated with the interference constraints.
    # First enforce the self-conflicts between frequencies in the same node:
    T = get_forbidden_set(1, 1, reuse_distances)
    for v in nodes:
        for f in range(nfreq):
            for g in range(f + 1, nfreq):
                if abs(f - g) in T:
                    var_i = f'x_{v}_{f}'
                    var_j = f'x_{v}_{g}'
                    bqm.add_interaction(var_i, var_j, penalty_coef)

    # Now enforce the cross-node conflicts:
    for iv, v in enumerate(nodes):
        for w in nodes[iv + 1:]:
            T = get_forbidden_set(v, w, reuse_distances)
            if not T:
                continue  # No disallowed frequencies at this distance
            for f in range(nfreq):
                for g in range(nfreq):
                    if abs(f - g) in T:
                        var_i = f'x_{v}_{f}'
                        var_j = f'x_{w}_{g}'
                        bqm.add_interaction(var_i, var_j, penalty_coef)

    return bqm

def bqm_to_numpy_qubo(bqm):
    """
    Convert a dimod.BinaryQuadraticModel to a symmetric QUBO matrix (numpy.ndarray)
    and return the offset separately.
    """
    # Convert the BQM to QUBO form (dictionary and offset)
    qubo_dict, offset = bqm.to_qubo()
    # Order the variables alphabetically (or any fixed order)
    variables = sorted(list(bqm.variables))
    n = len(variables)
    Q = np.zeros((n, n))
    # Fill in QUBO matrix from dictionary entries
    var_index = {v: i for i, v in enumerate(variables)}
    for (u, v), coeff in qubo_dict.items():
        i = var_index[u]
        j = var_index[v]
        Q[i, j] = coeff
        # Ensure symmetry; QDeepHybridSolver expects a symmetric matrix.
        if i != j:
            Q[j, i] = coeff
    return Q, offset, variables

if __name__ == '__main__':
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Run the frequency selection example on specified problem",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            The Philadelphia problem instances have the following minimum
            span frequency ranges:

            - P1: 426
            - P2: 426
            - P3: 257
            - P4: 252
            - P5: 239
            - P6: 179
            - P7: 855
            - P8: 524
            - P9: 1713

            In theory, each problem instance has a feasible solution when
            NFREQ is greater than or equal to the minimum span frequency
            range plus 1
            """))
    parser.add_argument("problem", nargs="?", default="small",
                        choices=["trivial", "single", "small", "very-small"] + ["P{}".format(i) for i in range(1,10)],
                        help="problem to run (default: %(default)s)")
    parser.add_argument('-n', '--nfreq', default=None, help="number of frequencies to consider (default: problem-dependent)", type=int)
    parser.add_argument('--show-frequencies', action="store_true", help="print out selected frequencies")
    parser.add_argument('--verbose', action='store_true', help='print details about frequency separation in solution (not allowed for full problem instances)')
    parser.add_argument("--show-plot",  action='store_true', help="display plot of cell grid")
    parser.add_argument("--save-plot",  action='store_true', help="save plot of cell grid to file")

    args = parser.parse_args()

    demand, nfreq, reuse_distances = load_problem(args.problem)
    if args.nfreq is not None:
        if args.nfreq <= 0:
            raise ValueError("number of frequencies must be positive")
        nfreq = args.nfreq

    print(nfreq, 'frequencies considered')

    bqm = construct_bqm(demand, nfreq, reuse_distances)
    print('{} variables'.format(bqm.num_variables))
    print('{} interactions'.format(bqm.num_interactions))
    
    # Convert the BQM to a numpy QUBO matrix for QDeepHybridSolver
    qubo_matrix, offset, variable_order = bqm_to_numpy_qubo(bqm)
    # (Note: the solver's energy will include the QUBO value; if desired, you may subtract the offset later.)

    # Initialize the QDeepHybridSolver and configure parameters
    solver = QDeepHybridSolver()
    # If an authentication token is needed, set it here:
    solver.token = "mtagdfsplb"
    solver.num_reads = 100

    # Solve the QUBO problem using the new hybrid solver
    try:
        response = solver.solve(qubo_matrix)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"API Error: {e}")
        exit(1)

    # The response contains a structured result; extract the solution configuration.
    # Expected response structure:
    # {
    #   "QdeepHybridSolver": {
    #       "configuration": [0.0, 1.0, ...],
    #       "energy": <float>,
    #       "time": <float>
    #   }
    # }
    sol_array = response["QdeepHybridSolver"]["configuration"]
    # Map the solution back to variable names based on our ordering:
    solution_dict = dict(zip(variable_order, sol_array))
    
    print("\nSolution:")
    # Check results using the provided utilities (same as original)
    violations = check_results(demand, nfreq, reuse_distances, solution_dict, verbose=False)
    print('{} demand violations'.format(violations['demand-count']))
    print('{} within-node frequency violations'.format(violations['self-count']))
    print('{} across-node frequency violations'.format(violations['cross-count']))
    print('')

    nodes = sorted(list(demand.keys()))
    frequencies = get_frequencies(nodes, nfreq, solution_dict)
    if args.show_frequencies:
        for node, f in sorted(frequencies.items()):
            print('Station {}: {}'.format(node, f))
        print('')
    station_maximums = [max(freqs) for freqs in frequencies.values() if freqs]
    if station_maximums:
        print('Max frequency:', max(station_maximums))

    if args.verbose:
        print('')
        print_frequency_separations(reuse_distances, frequencies)

    if args.show_plot or args.save_plot:
        interference = violations['self-nodes'].union(violations['cross-nodes'])
        plot_nodes(nodes, demand, interference, demand_violations=violations['demand-nodes'])

        if args.save_plot:
            filename = 'frequency_grid.png'
            plt.savefig(filename, bbox_inches='tight')
            print('Plot saved to:', filename)

        if args.show_plot:
            plt.show()
