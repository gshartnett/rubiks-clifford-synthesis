import sys

import numpy as np
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full

import rubiks.clifford as cl


def test_tableau_composition_no_phase_bits():
    """
    Check that two arbitrary tableaus (with the phase bit omitted)
    can be composed via matrix multiplication of the symplectic matrices
    """
    tableau1 = cl.Problem(5, seed=0).state
    tableau2 = cl.Problem(5, seed=10).state

    term1 = 1 * (tableau1 @ tableau2).tableau[:, :-1]

    term2 = np.dot(1 * tableau2.tableau[:, :-1], 1 * tableau1.tableau[:, :-1])
    term2 = term2 % 2

    assert np.array_equal(term1, term2)


def test_vectorized_tableau_composition():
    """
    Check that the composition of an arbitrary tableau with every tableau
    in the move set can be vectorized using numpy's einsum. This requires
    that the phase bits be dropped.
    """

    ## initialize the problem with a randomly chosen tableau
    problem = cl.Problem(5, drop_phase_bits=True)

    ## result using built-in Qiskit method
    arr1 = []
    for i in range(len(problem.move_set)):
        arr1.append(1 * problem.apply_move(problem.move_set[i]).tableau[:, :-1])
    arr1 = np.asarray(arr1)

    ## vectorized result
    arr2 = (
        np.einsum(
            "ij, mjk -> mik", 1 * problem.state.tableau[:, :-1], problem.move_set_array
        )
        % 2
    )

    ## check
    assert np.array_equal(arr1, arr2)


def test_hillclimbing_move():
    """
    This is yet another test of the matrix multiplication implementation of
    tableau composition (when the phase bits are omitted). This test was
    specifically introduced to make sure that the code used in the Hillclimbing
    functions had the correct order of tableau1 and tableau2 used in the
    composition.
    """
    problem = cl.Problem(5)

    print(problem.state.tableau.shape, problem.move_set_array.shape)

    ## compute the updated tableaus via vectorized approach (omitting the phase bits)
    candidates_1 = (
        np.einsum(
            "ij, mjk -> mik",
            1 * problem.state.tableau[:, :-1],
            problem.move_set_array[:, :, :-1],
        )
        % 2
    )

    candidates_2 = [
        tableau & problem.state for tableau in problem.move_set_tableau.values()
    ]
    # candidates_2 = [problem.state & tableau for tableau in problem.move_set_tableau.values()]
    candidates_2 = np.asarray([candidate.tableau[:, :-1] for candidate in candidates_2])
    print("yo")
    assert np.array_equal(candidates_1, candidates_2)


def test_move_set(num_trials=50, num_qubits_min=2, num_qubits_max=9):
    """
    Test that the move set agrees with the move set used by the
    Qiskit methods. Agreement is not necessary, but it does help
    facilitate a direct, oranges-to-oranges comparison.

    This test randomly generates Cliffords and then uses the
    Qiskit method to synthesize each circuit in terms of gates.

    Parameters
    ----------
    num_trials : int, optional
        Number of random trials, by default 50.
    num_qubits_min : int, optional
        Smallest number of qubits, by default 2.
    num_qubits_max : int, optional
        Largest number of qubits, by default 9.

    Raises
    ------
    NotImplementedError
        If a unexpected gate is encountered.
    """
    gate_set = set(["s", "sdg", "h", "x", "y", "z", "cx", "swap"])

    ## scan over trials
    for _ in range(num_trials):
        ## scan over qubits
        for n in range(num_qubits_min, num_qubits_max + 1):
            input_circuit = cl.random_clifford(n, 100, np.random.default_rng(1337))
            output_circuit = synth_clifford_full(Clifford(input_circuit))
            # sum(dict(input_circuit.count_ops()).values()), sum(dict(output_circuit.count_ops()).values())
            circ_gates = set(list(dict(output_circuit.count_ops()).keys()))
            try:
                assert circ_gates.issubset(gate_set)
            except:
                raise NotImplementedError("Gate sets do not match!")


def test_to_and_from_bitstring():
    """
    Check that a problem instance initialized from a bitstring agrees with the problem instance whose
    state corresponds to that same bitstring.
    """
    # keeping the phase bits
    problem1 = cl.Problem(num_qubits=5, drop_phase_bits=False)
    problem2 = cl.Problem(num_qubits=5, initial_state=problem1.to_bitstring())
    assert problem1.to_bitstring() == problem2.to_bitstring()

    # dropping the phase bits
    problem1 = cl.Problem(num_qubits=5, seed=4123)
    problem2 = cl.Problem(num_qubits=5, initial_state=problem1.to_bitstring(drop_phase_bits=True))
    assert np.array_equal(problem1.state.tableau[:,:-1], problem2.state.tableau[:,:-1])


def test_tableau_z_composition():
    '''
    Test the derived update rule for tableau composition with a
    Pauli Z:
        r_i \leftarrow r_i + x_{ia} (mod 2)
    '''
    problem = cl.Problem(num_qubits=6, seed=np.random.randint(low=100000000))
    initial_state = 1*problem.state.tableau

    qubit_idx = 0
    problem.apply_move((qubit_idx, 'z'), inplace=True)
    final_state = 1*problem.state.tableau

    # check that matrix parts of tableaus agree
    assert np.array_equal(initial_state[:,:-1], final_state[:,:-1])

    # check that the new phase bit obeys the derived relation
    for i in range(2*problem.num_qubits):
        ri_initial = initial_state[i, -1]

        xia_initial = initial_state[i, qubit_idx]
        zia_initial = initial_state[i, qubit_idx + problem.num_qubits]
        ri_final = final_state[i, -1]

        ri_final_computed = (ri_initial + xia_initial) % 2

        assert ri_final_computed == ri_final


def test_tableau_x_composition():
    '''
    Test the derived update rule for tableau composition with a
    Pauli X:
        r_i \leftarrow r_i + z_{ia} (mod 2)
    '''
    problem = cl.Problem(num_qubits=6, seed=np.random.randint(low=100000000))
    initial_state = 1*problem.state.tableau

    qubit_idx = 0
    problem.apply_move((qubit_idx, 'x'), inplace=True)
    final_state = 1*problem.state.tableau

    # check that matrix parts of tableaus agree
    assert np.array_equal(initial_state[:,:-1], final_state[:,:-1])

    # check that the new phase bit obeys the derived relation
    for i in range(2*problem.num_qubits):
        ri_initial = initial_state[i, -1]

        xia_initial = initial_state[i, qubit_idx]
        zia_initial = initial_state[i, qubit_idx + problem.num_qubits]
        ri_final = final_state[i, -1]

        ri_final_computed = (ri_initial + zia_initial) % 2

        assert ri_final_computed == ri_final