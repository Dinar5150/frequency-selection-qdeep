import numpy as np
import argparse
import textwrap

import matplotlib.pyplot as plt

from philadelphia import load_problem, get_forbidden_set, plot_nodes
from utilities import check_results, get_frequencies, print_frequency_separations

from qdeepsdk import QDeepHybridSolver

def construct_qubo(demand, nfreq, reuse_distances, penalty_coef=10.0):
    """
    Construct a QUBO matrix for the frequency assignment problem.
    Each binary variable x_v_f indicates whether frequency f is assigned to node v.

    Args:
        demand (dict): Mapping each node to its demand value.
        nfreq (int): Number of frequencies to consider.
        reuse_distances (list): List of reuse distances.
        penalty_coef (float): Penalty coefficient.

    Returns:
        tuple: (Q, var_index, index_var)
            Q (np.ndarray): The symmetric QUBO matrix.
            var_index (dict): Mapping from (node, frequency) to variable index.
            index_var (dict): Reverse mapping from index to (node, frequency).
    """
    nodes = sorted(list(demand.keys()))
    n_nodes = len(nodes)
    n_vars = n_nodes * nfreq
    Q = np.zeros((n_vars, n_vars))
    var_index = {}
    index_var = {}
    idx = 0
    for v in nodes:
        for f in range(nfreq):
            var_index[(v, f)] = idx
            index_var[idx] = (v, f)
            idx += 1

    # Constraint: Each node must meet its demand.
    # Linear part: add penalty_coef * (1 - 2*demand[v]) per variable.
    # Quadratic part: add penalty_coef * 2 for every pair (within a node).
    for v in nodes:
        for f in range(nfreq):
            i = var_index[(v, f)]
            Q[i, i] += penalty_coef * (1.0 - 2 * demand[v])
        for fi in range(nfreq):
            for fj in range(fi + 1, nfreq):
                i = var_index[(v, fi)]
                j = var_index[(v, fj)]
                Q[i, j] += penalty_coef * 2.0
                Q[j, i] = Q[i, j]  # ensure symmetry

    # Penalties for interference constraints:
    # 1. Self-conflicts within the same node.
    T_self = get_forbidden_set(1, 1, reuse_distances)
    for v in nodes:
        for f in range(nfreq):
            for g in range(f + 1, nfreq):
                if abs(f - g) in T_self:
                    i = var_index[(v, f)]
                    j = var_index[(v, g)]
                    Q[i, j] += penalty_coef
                    Q[j, i] = Q[i, j]

    # 2. Cross-node interference.
    for iv, v in enumerate(nodes):
        for w in nodes[iv + 1:]:
            T_cross = get_forbidden_set(v, w, reuse_distances)
            if not T_cross:
                continue
            for f in range(nfreq):
                for g in range(nfreq):
                    if abs(f - g) in T_cross:
                        i = var_index[(v, f)]
                        j = var_index[(w, g)]
                        Q[i, j] += penalty_coef
                        Q[j, i] = Q[i, j]
    return Q, var_index, index_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the frequency selection example on a specified problem using QDeepHybridSolver",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
            The Philadelphia problem instances have the following minimum span frequency ranges:

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
            range plus 1.
        """)
    )
    parser.add_argument("problem", nargs="?", default="small",
                        choices=["trivial", "single", "small", "very-small"] + ["P{}".format(i) for i in range(1, 10)],
                        help="Problem instance to run (default: %(default)s)")
    parser.add_argument('-n', '--nfreq', default=None, type=int,
                        help="Number of frequencies to consider (default: problem-dependent)")
    parser.add_argument('--show-frequencies', action="store_true", help="Print out selected frequencies")
    parser.add_argument('--verbose', action='store_true', help="Print details about frequency separation in the solution")
    parser.add_argument("--show-plot",  action='store_true', help="Display plot of cell grid")
    parser.add_argument("--save-plot",  action='store_true', help="Save plot of cell grid to file")
    args = parser.parse_args()

    # Load problem instance
    demand, nfreq, reuse_distances = load_problem(args.problem)
    if args.nfreq is not None:
        if args.nfreq <= 0:
            raise ValueError("Number of frequencies must be positive")
        nfreq = args.nfreq
    print(nfreq, 'frequencies considered')

    # Build the QUBO matrix from problem constraints.
    Q, var_index, index_var = construct_qubo(demand, nfreq, reuse_distances)
    n_vars = Q.shape[0]
    # Count nonzero upper-triangle entries as the number of interactions.
    interactions = np.count_nonzero(np.triu(Q, k=1))
    print(f"{n_vars} variables")
    print(f"{interactions} interactions")

    # Initialize the QDeepHybridSolver and set the authentication token.
    solver = QDeepHybridSolver()
    solver.token = "your-auth-token-here"  # Replace with your valid token

    try:
        # Solve the QUBO problem.
        result = solver.solve(Q)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"API Error: {e}")
        exit(1)

    # The API returns a dictionary with keys "configuration", "energy", and "time".
    solution_vector = result["configuration"]
    print("\nSolution:")

    # Check the solution using the same utility function.
    violations = check_results(demand, nfreq, reuse_distances, solution_vector, verbose=False)
    print(f"{violations['demand-count']} demand violations")
    print(f"{violations['self-count']} within-node frequency violations")
    print(f"{violations['cross-count']} across-node frequency violations")
    print("")

    nodes = sorted(list(demand.keys()))
    frequencies = get_frequencies(nodes, nfreq, solution_vector)
    if args.show_frequencies:
        for node, f in sorted(frequencies.items()):
            print(f"Station {node}: {f}")
        print("")
    if args.verbose:
        print("")
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
