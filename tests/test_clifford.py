import numpy as np
import sys
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full
import clifford as cl
#sys.path.append('..')
#import rubiks-clifford-synthesis.clifford as cl


def test_tableau_composition():
    """
    Check that two arbitrary tableaus (with the phase bit omitted)
    can be composed via matrix multiplication of the symplectic matrices
    """
    tableau1 = cl.Problem(5, seed=0).state
    tableau2 = cl.Problem(5, seed=10).state

    term1 = 1*(tableau1 @ tableau2).tableau[:,:-1]

    term2 = np.dot(1*tableau2.tableau[:,:-1], 1*tableau1.tableau[:,:-1])
    term2 = term2 % 2

    assert np.array_equal(term1, term2)


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
    gate_set = set(['s', 'sdg', 'h', 'x', 'y', 'z', 'cx', 'swap'])

    ## scan over trials
    for _ in range(num_trials):

        ## scan over qubits
        for n in range(num_qubits_min, num_qubits_max+1):
            input_circuit = cl.random_clifford(n, 100, np.random.default_rng(1337))
            output_circuit = synth_clifford_full(Clifford(input_circuit))
            #sum(dict(input_circuit.count_ops()).values()), sum(dict(output_circuit.count_ops()).values())
            circ_gates = set(list(dict(output_circuit.count_ops()).keys()))
            try:
                assert circ_gates.issubset(gate_set)
            except:
                raise NotImplementedError("Gate sets do not match!")