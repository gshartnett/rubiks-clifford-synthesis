from typing import (
    Dict, Union, List, Tuple
)
import os
import pickle
import argparse
from collections import Counter
from datetime import datetime
import tqdm
import numpy as np
import torch
from torch import nn, optim
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full
from qiskit import QuantumCircuit
from torch.utils.tensorboard import SummaryWriter


GATES = ['h', 's', 'sdg', 'x', 'y', 'z', 'cx', 'swap']


def normalize_dict(input_dict : Dict) -> Dict:
    """
    Normalize a dict whose keys are numbers.

    Parameters
    ----------
    input_dict : Dict
        An input dictionary of the form {key: number}

    Returns
    -------
    Dict
        A dictionary with normalized entries.
    """
    new_dict = input_dict.copy()
    total = sum(input_dict.values())
    for k in input_dict.keys():
        new_dict[k] /= total
    return new_dict


def bitstr_to_array(bitstr : str, num_qubits: int) -> np.ndarray:
    """
    Convert a bit string into a tableau, represented as a binary numpy
    array.

    Parameters
    ----------
    bitstr : str
        The input bitstring.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    np.ndarray
        A binary-valued numpy array.
    """
    assert len(bitstr) == (2*num_qubits) * (2*(num_qubits+1))
    arr = np.asarray([int(s) for s in bitstr]).reshape((2*num_qubits, 2*(num_qubits+1)))
    return arr


def array_to_bitstr(arr : np.ndarray) -> str:
    """
    Convert a binary numpy array into a bitstring.

    Parameters
    ----------
    arr : np.ndarray
        The array.

    Returns
    -------
    str
        The output string.
    """
    string = ''
    for arr_element in arr:
        string += str(arr_element)
    return string


def clifford_log_dim(
        num_qubits : int,
        qudit_dim : int = 2
        ) -> float:
    """
    Log dimension of the Clifford group, taken from here:
    https://quantumcomputing.stackexchange.com/questions/13643/is-the-clifford-group-finite
    p is the dimension of the qudit, assumed prime.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    qudit_dim : int, optional
        Qubit dimension, assumed prime. By default 2

    Returns
    -------
    float
        Log dimension of Clifford group.
    """
    out = (2*num_qubits)*np.log(qudit_dim) + (num_qubits**2)*np.log(qudit_dim)
    for i in range(1, num_qubits+1):
        out += np.log((qudit_dim**(2*i) - 1))
    return out


def size_movet_set(num_qubits : int) -> int:
    """
    Size of the move set. The move set is taken to be:
        Single qubit gates: X, Y, Z, H, S, Sdagger
        Two qubit gates: CNOT(i,j), CNOT(j,i), SWAP

    Parameters
    ----------
    num_qubits : int
        Number of qubits.

    Returns
    -------
    int
        Size of the move set.
    """
    num_single_qubit = 6*num_qubits
    num_CNOT = num_qubits*(num_qubits-1)
    num_SWAP = num_qubits*(num_qubits-1)/2
    return num_single_qubit + num_CNOT + num_SWAP


def random_sequence(
        rng : np.random._generator.Generator,
        seq_length : int,
        num_qubits: int
        ) -> List:
    """
    Generate a random gate sequence corresponding to num_qubits qubits.
    The length will either be seq_length or seq_length + 1, with the latter
    possibility occuring to enforce the fact that CNOT gates always act on
    two qubits, with one qubit being the target and one being the control.

    Parameters
    ----------
    rng : np.random._generator.Generator
        The pseudo-random number generator.
    seq_length : int
        Length of the sequence to be generated.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    List
        A list of moves defining the sequence.
    """
    seq = []
    for _ in range(seq_length):
        ## pick the qubit
        qubit1 = rng.choice(list(range(num_qubits)))

        ## pick the gate
        gate = rng.choice(GATES)

        ## if gate is one of the CNOT gates, add the complementary gate
        if gate in ['cx', 'swap',]:
            qubit2 = rng.choice(tuple(set(range(num_qubits)).difference({qubit1})))

            ## the SWAP gate is symmetric, so assume i < j
            if gate == 'swap':
                qubit1, qubit2 = np.sort([qubit1, qubit2])

            seq.append((qubit1, qubit2, gate))

        else:
            seq.append((qubit1, gate))

    return seq


def weighted_distance_to_identity(
        clifford_element : Union[QuantumCircuit, List],
        weight_dict : Dict,
        CNOTs_only : bool = False
        ) -> float:
    """
    Computed a weighted distance to the identity.
    The input variable can be either a sequence or a Qiskit circuit.
    Note that this is not necessarily the geodesic distance.

    Parameters
    ----------
    clifford_element : Union[QuantumCircuit, List]
        The Clifford element.
    weight_dict : Dict
        Edge weight dictionary of form {move : weight}.
    CNOTs_only : bool
        Boolean flag, if True then only count cx and swap gates.

    Returns
    -------
    float
        The weighted distance to the identity.
    """
    if CNOTs_only:
        weight_dict = {
            'cx':1,
            'swap':3
            }

    ## input is a circuit
    if isinstance(clifford_element, QuantumCircuit):
        gate_counts = dict(clifford_element.count_ops())
    ## input is a sequence of gates
    elif isinstance(clifford_element, list):
        gate_counts = Counter([item[-1] for item in clifford_element])
    else:
        raise NotImplementedError(
            'Input is assumed to be either a Qiskit QuantumCircuit or a list of moves'
            )
    weight = sum(weight_dict.get(key, 0)*value for key, value in gate_counts.items())
    return weight


def sequence_to_tableau(
        sequence : List,
        num_qubits : int
        ) -> Clifford:
    """
    Convert a sequence of Clifford gates into a tableau.

    Parameters
    ----------
    sequence : List
        A list of Clifford gates.
    num_qubits : int
        Number of qubits.

    Returns
    -------
    Clifford
        Clifford element/tableau.

    Raises
    ------
    NotImplementedError
        A warning in case a gate not in the gate set is encountered.
    """
    circ = QuantumCircuit(num_qubits)
    for operation in sequence:
        if operation[-1] == 'cx':
            circ.cx(int(operation[0]), int(operation[1]))
        elif operation[-1] == 'swap':
            circ.swap(int(operation[0]), int(operation[1]))
        elif operation[-1] == 's':
            circ.s(int(operation[0]))
        elif operation[-1] == 'sdg':
            circ.sdg(int(operation[0]))
        elif operation[-1] == 'h':
            circ.h(int(operation[0]))
        elif operation[-1] == 'x':
            circ.x(int(operation[0]))
        elif operation[-1] == 'y':
            circ.y(int(operation[0]))
        elif operation[-1] == 'z':
            circ.z(int(operation[0]))
        else:
            print('Error, gate not recognized!')
            print(operation)
            raise NotImplementedError
    return Clifford(circ)


def random_clifford(
        num_qubits : int,
        num_gates : int,
        rng : np.random._generator.Generator
        ) -> QuantumCircuit:
    """
    Generate a random Clifford circuit (not tableau) using the
    gate set (H, S, CNOT).
    Note that the distribution implied by this sampling procedure
    is NOT the uniform distribution. For uniform sampling, see
    https://qiskit.org/documentation/stubs/qiskit.quantum_info.random_clifford.html

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    num_gates : int
        Number of gates.
    rng : np.random._generator.Generator
        Pseudo-random number generator.

    Returns
    -------
    QuantumCircuit
        A Qiskit QuantumCircuit.
    """
    circ = QuantumCircuit(num_qubits)
    gate_set = ['h', 's', 'cx']
    for _ in range(num_gates):
        gate = gate_set[rng.choice(3)]
        if gate == 'h':
            circ.h(rng.choice(num_qubits))
        elif gate == 's':
            circ.s(rng.choice(num_qubits))
        else:
            i, j = rng.choice(num_qubits, size=2, replace=False)
            circ.cx(i, j)
    #return Clifford(circ) #convert the circuit to a tableau using the synthesis approach
    return circ


def pearson_correlation(
        input_sequence_1 : torch.Tensor,
        input_sequence_2 : torch.Tensor
        ):
    """
    The Pearson correlation coefficient.

    Parameters
    ----------
    input_sequence_1 : torch.Tensor
        Sequence 1.
    input_sequence_2 : torch.Tensor
        Sequence 2.

    Returns
    -------
    torch.Tensor
        The Pearson correlation coefficient of two sequences.
    """
    pearson_eps = 1e-10 #used to regulate correlation calculation

    delta_1 = input_sequence_1 - torch.mean(input_sequence_1, axis=0, keepdim=True)
    delta_2 = input_sequence_2 - torch.mean(input_sequence_2, axis=0, keepdim=True)
    corr = torch.sum(delta_1 * delta_2, dim=0, keepdim=True)
    corr = corr/(torch.sqrt(torch.sum(delta_1 ** 2 + pearson_eps, dim=0, keepdim=True)) \
        * torch.sqrt(torch.sum(delta_2 ** 2 + pearson_eps, dim=0, keepdim=True)))
    corr = torch.mean(corr)
    return corr


def generate_data_batch(
        rng : np.random._generator.Generator,
        num_batch : int,
        num_qubits : int,
        device : torch.device,
        weight_dict : Dict,
        high : Union[None, int] = None,
        drop_phase_bits : bool = False,
        use_qiskit : bool = False
        ) -> torch.tensor:
    """
    Generate a batch of tableaus, flattened into vectors,
    generated by sampling sequences and converting the
    sequences into tableaus.

    To Do: what's a good distribution for the lengths?

    Parameters
    ----------
    rng : np.random._generator.Generator
        Pseudo-random number generator.
    num_batch : int
        Number of batches to generate.
    num_qubits : int
        Number of qubits.
    device : torch.device
        PyTorch device.
    weight_dict : Dict
        Edge weight dictionary of form {move : weight}.
    high : Union[None, int], optional
        Maximum sequence length, by default 20*n.
    drop_phase_bits : bool, optional
        Whether the phase bits should be dropped, by default False.
    use_qiskit : bool, optional
        Whether the Qiskit built-in synthesis method should be used, by default False.

    Returns
    -------
    torch.tensor
        _description_
    """
    x_batch = []
    seq_lens = []
    normalized_weight_dict = normalize_dict(weight_dict)

    if high is None:
        high = 20 * np.log(num_qubits)/np.log(2)

    for _ in range(num_batch):
        ## sample a random sequence
        sequence_length = rng.integers(low=1, high=high)
        sequence = random_sequence(rng, sequence_length, num_qubits)
        distance_raw = weighted_distance_to_identity(sequence, normalized_weight_dict)
        tableau = sequence_to_tableau(sequence, num_qubits).tableau
        if drop_phase_bits:
            tableau = tableau[:,:-1]
        x_batch.append(1 * tableau.flatten())

        ## find the length returned by the synthesis approach
        ## this slows it down by quite a bit
        if use_qiskit:
            circ = sequence_to_tableau(sequence, num_qubits).to_circuit()
            distance_synthesis = weighted_distance_to_identity(circ, normalized_weight_dict)
        else:
            distance_synthesis = np.inf

        ## take the minimum
        seq_lens.append(min(distance_raw, distance_synthesis))

    x_batch = np.asarray(x_batch, dtype=np.float32)
    seq_lens = np.asarray(seq_lens, dtype=np.float32)

    return (
        torch.tensor(x_batch).to(device),
        torch.tensor(seq_lens).to(device)
        )


class Problem:
    """
    Problem class used to represent the current state
    and move set of the Clifford synthesis problem.
    """
    def __init__(
            self,
            num_qubits,
            initial_state=None,
            seed=123,
            high=None,
            drop_phase_bits=False
            ):

        self.num_qubits = num_qubits
        self.drop_phase_bits = drop_phase_bits
        self.rng = np.random.default_rng(seed)
        self.move_set = self.get_move_set()
        self.move_set_tableau = self.get_move_set_as_tableaus()
        self.move_set_array = self.get_move_set_as_array()
        if initial_state is not None:
            self.state = initial_state
        else:
            if high is None:
                high = int(20*np.log(num_qubits)/np.log(2))
            seq_len = self.rng.integers(low=1, high=high)
            self.state = sequence_to_tableau(
                random_sequence(self.rng, seq_len, num_qubits), num_qubits
                )

    def get_move_set(self) -> List:
        """
        Generate the move set for the n-qubit problem.

        Returns
        -------
        List
            The move set.
        """
        move_set = []
        for i in range(self.num_qubits):
            move_set.append((i, 'h'))
            move_set.append((i, 's'))
            move_set.append((i, 'sdg'))
            move_set.append((i, 'x'))
            move_set.append((i, 'y'))
            move_set.append((i, 'z'))
            if i != self.num_qubits - 1:
                for j in range(i+1,self.num_qubits):
                    move_set.append((i, j, 'cx'))
                    move_set.append((j, i, 'cx'))
                    move_set.append((i, j, 'swap'))
                    #move_set.append((j, i, 'SWAP'))
        return move_set

    def get_move_set_as_tableaus(self) -> Dict:
        """
        Generate the moves as individual tableaus.

        Returns
        -------
        Dict
            A dictionary of the form {move: tableau}
        """
        return {
            self.move_set[i] : sequence_to_tableau(
                [self.move_set[i]], self.num_qubits) for i in range(len(self.move_set))
            }

    def get_move_set_as_array(self) -> np.ndarray:
        """
        Generate the moves as a an (M, 2n, 2n + 1) array, with M the number of
        moves.

        Returns
        -------
        np.ndarray
            The moves represented as an array.
        """
        num_moves = len(self.move_set)
        if self.drop_phase_bits:
            move_arr = np.zeros((num_moves, 2*self.num_qubits, 2*self.num_qubits))
            for i_move in range(num_moves):
                move = self.move_set[i_move]
                move_arr[i_move, :, :] = 1*self.move_set_tableau[move].tableau[:,:-1]
        else:
            move_arr = np.zeros((num_moves, 2*self.num_qubits, 2*self.num_qubits + 1))
            for i_move in range(num_moves):
                move = self.move_set[i_move]
                move_arr[i_move, :, :] = 1*self.move_set_tableau[move].tableau
        return move_arr

    def apply_move(
            self,
            move : str,
            inplace : bool = False
            ) -> Union[None, Clifford]:
        """
        Apply a move to the problem state.
        The composition operation corresponds to appending the move to
        the end of the circuit, see
        https://qiskit.org/documentation/stubs/qiskit.quantum_info.Clifford.compose.html#qiskit.quantum_info.Clifford.compose

        Parameters
        ----------
        move : str
            The move to be applied
        inplace : bool, optional
            Controls whether the move should be made in place, by default False.

        Returns
        -------
        Union[None, Clifford]
            Returns None if inplace, otherwise return the new state.
        """
        ## THIS SHOULD USE drop_phase_bits
        if not inplace:
            return self.state & self.move_set_tableau[move]
        else:
            self.state = self.state & self.move_set_tableau[move]
            return None

    def random_move(self, inplace : bool = False) -> Tuple:
        """
        Apply a random move.

        TO DO:
        Add the option for a user-defined probability distribution

        Parameters
        ----------
        inplace : bool, optional
            Controls whether the move should be made in place, by default False.

        Returns
        -------
        Tuple
            A tuple of the move and the tableau (which might be none if the move is made in place).
        """
        move = self.move_set[self.rng.choice(len(self.move_set))]
        return move, self.apply_move(move, inplace=inplace)

    def is_solution(self) -> bool:
        """
        Checks whether the current problem state is the identity,
        corresponding to the solved tableau.

        Returns
        -------
        bool
            Whether the current state is the identity tableau.
        """
        if self.drop_phase_bits:
            if np.array_equal(1*self.state.tableau[:,:-1], np.eye(2*self.num_qubits)):
                return True
        else:
            if np.array_equal(1*self.state.tableau, sequence_to_tableau([], self.num_qubits)):
                return True
        return False

    def find_move_from_a_to_b(
            self,
            tableau_a : Clifford,
            tableau_b : Clifford
            ) -> Union[None, str]:
        """
        Find the move, if it exists, that maps A to B:
        B = A & move
        move = A^{-1} & B

        Parameters
        ----------
        tableau_a : Clifford
            First Clifford element.
        tableau_b : Clifford
            Second Clifford element.

        Returns
        -------
        Union[None, str]
            None if there is no move connecting A and B, otherwise the move string.
        """
        for move, tableau in self.move_set_tableau.items():
            if (tableau_a.adjoint() & tableau_b) == tableau:
                return move
        return None


def hillclimbing_random(
        initial_state : Clifford,
        max_iter : int = 1000,
        seed : int = 123,
        ) -> Dict:
    """
    A simple random walk algorithm for solving the synthesis problem.
    The main reason why this algorithm is of interest is to check that
    the success probability of the other algorithms is above this simple
    baseline.

    Parameters
    ----------
    initial_state : Clifford
        The initial Clifford state.
    max_iter : int, optional
        Maximum number of iterations, by default 1000.
    seed : int, optional
        Pseudorandom number generator seed, by default 123.

    Returns
    -------
    Dict
        A dictionary containing the result.
    """

    ## initialize the problem
    problem = Problem(
        initial_state.tableau.shape[0]//2,
        initial_state=initial_state.copy(),
        seed=seed
        )
    identity_tableau = sequence_to_tableau([], problem.num_qubits).tableau
    result = {'success':False, 'move_history':[]}

    for _ in range(max_iter):

        ## check for solution
        if np.array_equal(problem.state.tableau, identity_tableau):
            result['success'] = True
            return result

        ## make a random move
        move, _ = problem.random_move()
        result['move_history'].append(move)
        problem.apply_move(move, inplace=True)

    return result


def hillclimbing(
        initial_state : Clifford,
        lgf_model : nn.Module,
        max_iter : int = 1000,
        seed : int = 123,
        ) -> Dict:
    """
    The hillclimbing algorithm used to unscramble tableaus.

    TO DO:
    Should this be implemented as it is currently, or joined
    in some way with either the problem class or the LGF class?

    Parameters
    ----------
    initial_state : Clifford
        The initial state.
    lgf_model : nn.Module
        The learned guidance function.
    max_iter : int, optional
        The maximum number of iterations, by default 1000.
    seed : int, optional
        The pseudo-random number generator seed, by default 123.

    Returns
    -------
    Dict
        A result dictionary.
    """
    assert initial_state.tableau.shape[0]//2 == lgf_model.num_qubits
    result = {'success':False, 'move_history':[], 'L_history':[], 'move_type':[]}

    ## initialize
    problem = Problem(
        lgf_model.num_qubits,
        initial_state=initial_state.copy(),
        seed=seed,
        drop_phase_bits=lgf_model.drop_phase_bits
        )
    num_moves = len(problem.move_set)
    lgf_value = np.inf

    ## build the identity tableau (as a numpy array)
    if problem.drop_phase_bits:
        identity = np.eye(2*problem.num_qubits)
    else:
        identity = 1 * Clifford(QuantumCircuit(lgf_model.num_qubits)).tableau

    ## check for solution
    if problem.is_solution():
        result['success'] = True
        return result

    ## loop over iterations
    for _ in range(max_iter):

        ## build array of candidates
        if lgf_model.drop_phase_bits:
            candidates = np.einsum(
                'ij, mjk -> mik',
                1*problem.state.tableau[:,:-1],
                problem.move_set_array
                ) % 2
        else:
            """
            This is quite slow. I did some profiling experiments and confirmed that the bottleneck is
            the first line where the tableau composition is performed, and not the casting to a numpy
            array.

            To improve this, we would need to implement a vectorized version of the `_compose_general`
            classmethod defined here:
            https://qiskit.org/documentation/_modules/qiskit/quantum_info/operators/symplectic/clifford.html#Clifford.compose
            """
            candidates = [problem.state & tableau for tableau in problem.move_set_tableau.values()]
            #candidates = [tableau & problem.state for tableau in problem.move_set_tableau.values()]
            candidates = np.asarray([candidate.tableau for candidate in candidates])

        ## check to see if any of the candidates are the solution
        for i_move in range(num_moves):
            if problem.drop_phase_bits:

                ## confirm the array shape is correct
                assert candidates.shape == (
                    len(problem.move_set),
                    2*problem.num_qubits,
                    2*problem.num_qubits
                    )

                if np.array_equal(candidates[i_move], identity):
                    result['move_history'].append(problem.move_set[i_move])
                    result['move_type'].append('Solution')
                    result['success'] = True
                    with torch.no_grad():
                        result['L_history'].append(lgf_model.forward(torch.tensor(
                            candidates[i_move].flatten()[None,:], dtype=torch.float32
                            )))
                    return result
            else:

                ## confirm the array shape is correct
                assert candidates.shape == (
                    len(problem.move_set),
                    2*problem.num_qubits,
                    2*problem.num_qubits + 1
                    )

                if np.array_equal(candidates[i_move], identity):
                    result['move_history'].append(problem.move_set[i_move])
                    result['move_type'].append('Solution')
                    result['success'] = True
                    with torch.no_grad():
                        result['L_history'].append(
                            lgf_model.forward(torch.tensor(
                            candidates[i_move].flatten()[None,:], dtype=torch.float32)
                                )
                            )
                    return result

        ## evaluate LGF for each candidate
        candidates = torch.tensor(candidates, dtype=torch.float32)
        candidates = torch.flatten(candidates, start_dim=1)
        with torch.no_grad():
            lgf_of_candidates = lgf_model.forward(candidates).numpy()

        ## pick the best candidate (with random tie breaking)
        #i_best = np.argmin(lgf_of_candidates)
        i_best = np.random.choice(
            np.flatnonzero(
                np.isclose(lgf_of_candidates, lgf_of_candidates.min())
                )
            )
        lgf_best = lgf_of_candidates[i_best]

        ## apply move (best or random)
        if lgf_best < lgf_value:
            lgf_value = lgf_best
            result['L_history'].append(lgf_value)
            result['move_history'].append(problem.move_set[i_best])
            result['move_type'].append('LGF')
            problem.apply_move(problem.move_set[i_best], inplace=True)
        else:
            i_move = lgf_model.rng.choice(len(problem.move_set))
            move = problem.move_set[i_move]
            lgf_value = lgf_of_candidates[i_move]
            result['L_history'].append(lgf_value)
            result['move_history'].append(move)
            result['move_type'].append('Random')
            problem.apply_move(move, inplace=True)

    return result


def evolutionary_strategy(
        initial_state : Clifford,
        lgf_model : nn.Module,
        population_size : int = 10,
        max_iter : int = 1000,
        seed : int = 123,
        ) -> Dict:
    assert initial_state.tableau.shape[0]//2 == lgf_model.num_qubits
    result = {'success':False, 'move_history':[], 'L_history':[], 'move_type':[]}

    ## create initial population
    current_population = [
        Problem(
        num_qubits = lgf_model.num_qubits,
          initial_state = initial_state
          )
        for _ in range(population_size)]

    ## loop over iterations
    for iter in range(max_iter):

        ## create mutated population (P')
        mutant_population = []
        for i in range(population_size):
            mutant = current_population[i].random_move(inplace=False)[1]
            if mutant.is_solution():
                print('solved it!')
                return
            else:
                mutant_population.append(mutant)

        ## evaluate LGF for each mutant
        mutant_state_array = [mutant.tableau[:,:-1] for mutant in mutant_population]
        mutant_state_array = torch.tensor(mutant_state_array, dtype=torch.float32)
        mutant_state_array = torch.flatten(mutant_state_array, start_dim=1)
        with torch.no_grad():
            lgf_of_mutants = lgf_model.forward(mutant_state_array).numpy()

        ## create population consisting of improved mutants only
        strong_mutant_population = []

        for i in range(population_size):
            x = 0
            # do something here!

    return result


def compute_weighted_steps_until_success(
        lgf_model : nn.Module,
        weight_dict : Dict,
        num_trials : int = 100,
        max_iter : int = int(1e4),
        method : str = 'lgf'
        ) -> Dict:
    """
    Apply the hillclimbing algorithm a number of times to find the
    distribution of steps until success.

    Parameters
    ----------
    lgf_model : nn.Module
        Learned guidance function.
    weight_dict : Dict
        Edge weight dictionary of form {move : weight}.
    num_trials : int, optional
        Number of trials to consider, by default 100.
    max_iter : int, optional
        Maximum number of iterations, by default 10,000.
    method : str, optional
        Method to use, by default 'lgf'.

    Returns
    -------
    Dict
        Two lists:
            - a list of steps until success for each trial.
            - a list of CNOTs until success for each trial.

    Raises
    ------
    ValueError
        Raise a value error if the method is not recognized.
    """
    ## loop over many repetitions of the hillclimbing algorithm
    steps_until_success = []
    CNOT_count = []
    for k in tqdm.trange(num_trials):

        problem = Problem(lgf_model.num_qubits, seed=k)

        ## apply the synthesis method
        if method == 'lgf':
            result = hillclimbing(
                problem.state,
                lgf_model,
                max_iter=max_iter,
                seed=k,
                )
        elif method == 'random':
            result = hillclimbing_random(
                problem.state,
                max_iter=max_iter,
                seed=k,
                )
        elif method == 'qiskit':
            ## hacky way to put the Qiskit results in the same data structure as the above
            result = {
                'success' : True,
                'move_history' : synth_clifford_full(problem.state)
                }
        else:
            raise NotImplementedError('synthesis method not recognized')

        ## compute the weighted distance and CNOT count
        if result['success']:
            steps_until_success.append(
                weighted_distance_to_identity(result['move_history'], weight_dict)
                )
            CNOT_count.append(
                weighted_distance_to_identity(
                    result['move_history'],
                    weight_dict,
                    CNOTs_only=True
                    )
                )
        else:
            steps_until_success.append(None)
            CNOT_count.append(None)

    return {'weighted_steps':steps_until_success, 'CNOTs':CNOT_count}


def conv2d_output_dimensions(
        height_input : int,
        width_input : int,
        kernel_size : int,
        stride : int,
        dilation : int = 1,
        padding : int = 0
        ) -> Tuple[int, int]:
    """
    Function used to determine how the shape of a tensor changes after
    the forward pass of a conv2d layer.
    See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Parameters
    ----------
    height_input : int
        The H channel dimension of the input.
    width_input : int
        The W channel dimension of the input.
    kernel_size : int
        The kernel size.
    stride : int
        The stride.
    dilation : int, optional
        The dilation parameter, by default 1.
    padding : int, optional
        The padding parameter, by default 0.

    Returns
    -------
    Tuple[int, int]
        The transformed H and W channel dimensions.
    """
    height_output = (height_input + 2*padding - dilation*(kernel_size - 1) - 1) / (stride) + 1
    width_output = (width_input + 2*padding - dilation*(kernel_size - 1) - 1) / (stride) + 1
    return int(height_output), int(width_output)


class LGFModel(nn.Module):
    """
    Learned Guidance Function model.
    """
    def __init__(
            self,
            num_qubits,
            device,
            rng,
            hidden_layers=[32, 16, 4],
            drop_phase_bits = False,
            use_qiskit = False
            ):
        super(LGFModel, self).__init__()
        self.num_qubits = num_qubits
        self.drop_phase_bits = drop_phase_bits
        self.use_qiskit = use_qiskit # can remove support

        if drop_phase_bits:
            self.input_dim = 2*num_qubits * (2*num_qubits)
        else:
            self.input_dim = 2*num_qubits * (2*num_qubits+1)

        self.hidden_layers =[self.input_dim] + hidden_layers + [1]
        self.device = device
        self.rng = rng
        self.fc_layers = nn.ModuleList(
            [nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]
            ) for i in range(len(self.hidden_layers)-1)]
        )
        ## nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        #self.conv2d_1 = nn.Conv2d(1, 1, 2, stride=1)
        #self.conv2d_2 = nn.Conv2d(1, 1, 2, stride=1) # can either use or remove support

        #H_input = 2*self.num_qubits + 1
        #W_input = 2*self.num_qubits
        #H_output, W_output = conv2d_output_dimensions(H_input, W_input, 2, 1)
        #H_output, W_output = conv2d_output_dimensions(H_output, W_output, 2, 1)
        #self.fc1 = nn.Linear(H_output * W_output, 1)

    def forward(self, x):
        """
        Forward pass
        """
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i != len(self.fc_layers) - 1:
                x = nn.Sigmoid()(x)
            x = torch.exp(x)
        return x.squeeze()

    def eval(self, x: Clifford):
        """
        A simple wrapper for the forward pass which accepts
        inputs formatted as tableaus, as opposed to flattened
        pytorch tensors.
        """
        assert isinstance(x, Clifford)
        if self.drop_phase_bits:
            x_tensor = torch.tensor(1.0 * x.tableau.flatten()[None,:-1], dtype=torch.float32)
        else:
            x_tensor = torch.tensor(1.0 * x.tableau.flatten()[None,:], dtype=torch.float32)
        with torch.no_grad():
            return self.forward(x_tensor).item()

    def train(
            self,
            batch_size : int,
            learning_rate : float,
            num_epochs : int,
            weight_dict : Dict,
            high : Union[None, int] = None
            ) -> List:
        """
        Train the LGF.

        Parameters
        ----------
        batch_size : int
            The batch size.
        learning_rate : float
            The learning rate.
        num_epochs : int
            Number of epochs for the training.
        weight_dict : Dict
            Edge weight dictionary of form {move : weight}.
        high : Union[None, int]
            Maximum random sequence length, by default None.

        Returns
        -------
        List
            List of loss values after each epoch.
        """
        _loss_history = []
        date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        writer = SummaryWriter(f'runs/date_{date}_num_qubits_{n}')
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #scheduler = optim.lr_scheduler.OneCycleLR(
        #    optimizer,
        #    max_lr=1e-3,
        #    steps_per_epoch=1,
        #    epochs=num_epochs
        #    )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            )
        loss_current = np.inf

        ## loop over "epochs"
        pbar = tqdm.trange(num_epochs)
        for epoch in pbar:

            ## update the progress bar
            pbar.set_description(f'loss={loss_current:.4f}, lr={learning_rate:.3e}')

            ## sample a batch of Cliffords
            x_batch, seq_lens = generate_data_batch(
                self.rng,
                batch_size,
                self.num_qubits,
                self.device,
                weight_dict,
                high,
                self.drop_phase_bits,
                self.use_qiskit
                )
            out = self.forward(x_batch)

            #loss = torch.nn.MSELoss(reduction='mean')(out, seq_lens)
            #loss = torch.mean(torch.square(out - seq_lens))
            loss = - pearson_correlation(out, seq_lens)

            writer.add_scalar("loss", loss, epoch)
            writer.add_scalar("lr", learning_rate, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]

            #loss_current = loss.detach().numpy().item()
            loss_current = loss.item()
            _loss_history.append(loss_current)

        writer.close()
        return _loss_history


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train the Rubiks-inspired approach for Clifford syntehsis.'
    )

    parser.add_argument(
        '--num_qubits',
        default=5,
        type=int,
        help='number of qubits'
    )

    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='learning rate'
    )

    parser.add_argument(
        '--batch_size',
        default=2000,
        type=int,
        help='batch size'
    )

    parser.add_argument(
        '--num_epochs',
        default=500,
        type=int,
        help='number of epochs'
    )

    parser.add_argument(
        '--eval_num_trials',
        default=2000,
        type=int,
        help='number of evaluation trials'
    )

    parser.add_argument(
        '--eval_max_iter',
        default=1000,
        type=int,
        help='maximum number of iterations for evaluation'
    )

    parser.add_argument(
        '--use_gpu',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Use the GPU, if available'
        )

    parser.add_argument(
        '--use_qiskit',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Boost the training by using the qiskit decomposition'
        )

    parser.add_argument(
        '--drop_phase_bits',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Drop the phase bits of the tableau'
        )

    parser.add_argument(
        '--load_saved_model',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Load a previously saved model, if available'
        )

    args = vars(parser.parse_args())

    print('Run parameters\n==================================')
    for key, value in args.items():
        print(f'{key}: {value}')

    ## if there is a GPU, use it
    device = torch.device("cuda" if (args['use_gpu'] and torch.cuda.is_available()) else "cpu")
    #device = torch.device('cpu')
    print(f'\nRunning with device={device}\n')

    ## save directory
    n = args['num_qubits']
    if args['drop_phase_bits']:
        data_dir = f'data/data_n_{n}_drop_phase_bits/'
    else:
        data_dir = f'data/data_n_{n}/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ## run hyper-parameters
    WEIGHT_DICT = {
        'x':1,
        'y':1,
        'z':1,
        'h':1,
        's':1,
        'sdg':1,
        'cx':30,
        'swap':90
        }
    use_qiskit = args['use_qiskit']
    SEED = 123

    ## iniitialize model
    lgf_model = LGFModel(
        num_qubits=args['num_qubits'],
        device=device,
        rng=np.random.default_rng(SEED),
        hidden_layers=[32, 16, 4, 2],
        drop_phase_bits=args['drop_phase_bits'],
        use_qiskit=use_qiskit
        ).to(device)

    ## either load previously trained model or train from scratch
    if args['load_saved_model'] and os.path.exists(data_dir + 'checkpoint'):
        lgf_model.load_state_dict(torch.load(data_dir + 'checkpoint'))
    else:
        ## train model
        print('training model:')
        loss_history = lgf_model.train(
            batch_size=args['batch_size'],
            lr=args['learning_rate'],
            num_epochs=args['num_epochs'],
            weight_dict=WEIGHT_DICT,
            high=None #not sure what a good choice is here
            )

    ## evaluate steps until success
    if True:

        steps_until_success = {}

        ## lgf hillclimbing
        print('evaluating hillclimbing using LGF')
        steps_until_success['lgf'] = compute_weighted_steps_until_success(
            lgf_model=lgf_model,
            weight_dict=WEIGHT_DICT,
            num_trials=args['eval_num_trials'],
            max_iter=args['eval_max_iter'],
            method='lgf'
            )

        ## qiskit method
        steps_until_success['qiskit'] = compute_weighted_steps_until_success(
            lgf_model=lgf_model,
            weight_dict=WEIGHT_DICT,
            num_trials=args['eval_num_trials'],
            max_iter=args['eval_max_iter'],
            method='qiskit'
            )

    ## save
    ## model weights
    if not (args['load_saved_model'] and os.path.exists(data_dir + 'checkpoint')):
        torch.save(lgf_model.state_dict(), data_dir + 'checkpoint')

        ## loss history
        np.save(data_dir + 'loss_history', loss_history)

        ## command line args
        with open(data_dir + 'args.pkl', 'wb') as f:
            pickle.dump(args, f)

        ## gate weight dictionary
        with open(data_dir + 'weight_dict.pkl', 'wb') as f:
            pickle.dump(WEIGHT_DICT, f)

    ## evaluation results
    #if args['evaluate']:
    with open(data_dir + 'steps_until_success.pkl', 'wb') as f:
        pickle.dump(steps_until_success, f)
