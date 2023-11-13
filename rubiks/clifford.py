from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from numpy.random._generator import Generator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info import random_clifford as random_clifford_uniform

GATES = ["h", "s", "sdg", "x", "y", "z", "cx", "swap"]


class Node:
     # Node class for linked list
	def __init__(self, data):
		self.data = data
		self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
 
    def insertAtBegin(self, data):
        # add a node at begin of linked list
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        else:
            new_node.next = self.head
            self.head = new_node

    def inserAtEnd(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        current_node = self.head
        while(current_node.next):
            current_node = current_node.next    
        current_node.next = new_node
        
    def get_move_list(self):
        # make a pass through the list and retrieve the moves
        move_list = []
        current_node = self.head
        while(current_node):
            move_list.append(current_node.data['move'])
            current_node = current_node.next
        return move_list
    
    def copy(self):
        new_list = LinkedList()
        buffer = self.head
        while buffer.next != None:
            new_list.inserAtEnd(buffer.data)
            buffer= buffer.next
        new_list.inserAtEnd(buffer.data)
        return new_list


def pad_bitstring(bitstring_dropped, num_qubits):
    """
    Given a bitstring representation of a tableau with the phase bits dropped,
    add in the phase bits (set to 0).

    Parameters
    ----------
    bitstring_dropped : _type_
        _description_
    num_qubits : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if len(bitstring_dropped) != (2*num_qubits) ** 2:
        raise ValueError("Input bitstring should have length (2n)^2")
    new_str = ''
    for i in range(2*num_qubits):
        new_str += bitstring_dropped[i*(2*num_qubits):(i+1)*(2*num_qubits)]
        new_str += '0'
    return new_str


def max_random_sequence_length(num_qubits: int, scaling: str) -> int:
    """
    Generate the high parameter used to set the maximum length of the
    random sequence.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    scaling : str
        Scaling type, can be log, linear, or log-linear.

    Returns
    -------
    int
        The high parameter.
    """
    if not scaling in ["log", "linear", "log-linear"]:
        raise ValueError("scaling must be one of: log, linear, log-linear")
    if scaling == "log":
        return int(20 * np.log(num_qubits) / np.log(2))
    elif scaling == "linear":
        return int(20 * num_qubits / 2)
    return int(10 * num_qubits * np.log(num_qubits) / np.log(2))


def normalize_dict(input_dict: Dict) -> Dict:
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


def TODELETE_bitstr_to_array(bitstr: str, num_qubits: int) -> np.ndarray:
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
    if not len(bitstr) == (2 * num_qubits) * (2 * num_qubits + 1):
        raise ValueError("The bitstring should have dimension 2n*(2n+1).")
    arr = np.asarray([int(s) for s in bitstr]).reshape(
        (2 * num_qubits, 2 * num_qubits + 1)
    )
    return arr


def TODELETE_array_to_bitstr(arr: np.ndarray) -> str:
    """
    Convert a binary numpy array into a bitstring.
    The array should be 1D.

    Parameters
    ----------
    arr : np.ndarray
        The array.

    Returns
    -------
    str
        The output string.
    """
    string = ""
    for arr_element in arr:
        string += str(arr_element)
    return string


def clifford_log_dim(num_qubits: int, qudit_dim: int = 2) -> float:
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
    out = (2 * num_qubits) * np.log(qudit_dim) + (num_qubits**2) * np.log(qudit_dim)
    for i in range(1, num_qubits + 1):
        out += np.log((qudit_dim ** (2 * i) - 1))
    return out


def size_movet_set(num_qubits: int) -> int:
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
    num_single_qubit = 6 * num_qubits
    num_CNOT = num_qubits * (num_qubits - 1)
    num_SWAP = num_qubits * (num_qubits - 1) / 2
    return num_single_qubit + num_CNOT + num_SWAP


def random_sequence(rng: Generator, seq_length: int, num_qubits: int) -> List:
    """
    Generate a random gate sequence corresponding to num_qubits qubits.
    The length will either be seq_length or seq_length + 1, with the latter
    possibility occuring to enforce the fact that CNOT gates always act on
    two qubits, with one qubit being the target and one being the control.

    Parameters
    ----------
    rng : Generator
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
        if gate in [
            "cx",
            "swap",
        ]:
            qubit2 = rng.choice(tuple(set(range(num_qubits)).difference({qubit1})))

            ## the SWAP gate is symmetric, so assume i < j
            if gate == "swap":
                qubit1, qubit2 = np.sort([qubit1, qubit2])

            seq.append((qubit1, qubit2, gate))

        else:
            seq.append((qubit1, gate))

    return seq


def weighted_distance_to_identity(
    clifford_element: Union[QuantumCircuit, List],
    weight_dict: Dict,
    CNOTs_only: bool = False,
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
        weight_dict = {"cx": 1, "swap": 3}

    ## input is a circuit
    if isinstance(clifford_element, QuantumCircuit):
        gate_counts = dict(clifford_element.count_ops())
    ## input is a sequence of gates
    elif isinstance(clifford_element, list):
        gate_counts = Counter([item[-1] for item in clifford_element])
    else:
        raise NotImplementedError(
            "Input is assumed to be either a Qiskit QuantumCircuit or a list of moves"
        )
    weight = sum(weight_dict.get(key, 0) * value for key, value in gate_counts.items())
    return weight


def sequence_to_tableau(sequence: List, num_qubits: int) -> Clifford:
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
        if operation[-1] == "cx":
            circ.cx(int(operation[0]), int(operation[1]))
        elif operation[-1] == "swap":
            circ.swap(int(operation[0]), int(operation[1]))
        elif operation[-1] == "s":
            circ.s(int(operation[0]))
        elif operation[-1] == "sdg":
            circ.sdg(int(operation[0]))
        elif operation[-1] == "h":
            circ.h(int(operation[0]))
        elif operation[-1] == "x":
            circ.x(int(operation[0]))
        elif operation[-1] == "y":
            circ.y(int(operation[0]))
        elif operation[-1] == "z":
            circ.z(int(operation[0]))
        else:
            print("Error, gate not recognized!")
            print(operation)
            raise NotImplementedError
    return Clifford(circ)


def random_clifford(num_qubits: int, num_gates: int, rng: Generator) -> QuantumCircuit:
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
    rng : Generator
        Pseudo-random number generator.

    Returns
    -------
    QuantumCircuit
        A Qiskit QuantumCircuit.
    """
    circ = QuantumCircuit(num_qubits)
    gate_set = ["h", "s", "cx"]
    for _ in range(num_gates):
        gate = gate_set[rng.choice(3)]
        if gate == "h":
            circ.h(rng.choice(num_qubits))
        elif gate == "s":
            circ.s(rng.choice(num_qubits))
        else:
            i, j = rng.choice(num_qubits, size=2, replace=False)
            circ.cx(i, j)
    # return Clifford(circ) #convert the circuit to a tableau using the synthesis approach
    return circ


def generate_data_batch(
    rng: Generator,
    num_batch: int,
    num_qubits: int,
    device: torch.device,
    weight_dict: Dict,
    high: Union[None, int] = None,
    drop_phase_bits: bool = False,
    use_qiskit: bool = False,
) -> torch.tensor:
    """
    Generate a batch of tableaus, flattened into vectors,
    generated by sampling sequences and converting the
    sequences into tableaus.

    To Do: what's a good distribution for the lengths?

    Parameters
    ----------
    rng : Generator
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
        high = 20 * np.log(num_qubits) / np.log(2)

    for _ in range(num_batch):
        ## sample a random sequence
        sequence_length = rng.integers(low=1, high=high)
        sequence = random_sequence(rng, sequence_length, num_qubits)
        distance_raw = weighted_distance_to_identity(sequence, normalized_weight_dict)
        tableau = sequence_to_tableau(sequence, num_qubits).tableau
        if drop_phase_bits:
            tableau = tableau[:, :-1]
        x_batch.append(1 * tableau.flatten())

        ## find the length returned by the synthesis approach
        ## this slows it down by quite a bit
        if use_qiskit:
            circ = sequence_to_tableau(sequence, num_qubits).to_circuit()
            distance_synthesis = weighted_distance_to_identity(
                circ, normalized_weight_dict
            )
        else:
            distance_synthesis = np.inf

        ## take the minimum
        seq_lens.append(min(distance_raw, distance_synthesis))

    x_batch = np.asarray(x_batch, dtype=np.float32)
    seq_lens = np.asarray(seq_lens, dtype=np.float32)

    return (torch.tensor(x_batch).to(device), torch.tensor(seq_lens).to(device))


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
        drop_phase_bits=False,
        sampling_method="random_walk",
    ):
        self.num_qubits = num_qubits
        self.drop_phase_bits = drop_phase_bits
        self.rng = np.random.default_rng(seed)
        self.move_set = self.get_move_set()
        self.move_set_tableau = self.get_move_set_as_tableaus()
        self.move_set_array = self.get_move_set_as_array()
        if initial_state is not None:
            if type(initial_state) is str:
                if len(initial_state) == (2*num_qubits) * (2*num_qubits + 1):
                    initial_state = np.array(list(initial_state), dtype=np.int).reshape((2*self.num_qubits, 2*self.num_qubits+1))
                    self.state = Clifford(initial_state)
                elif len(initial_state) == (2*num_qubits) * (2*num_qubits):
                    initial_state = pad_bitstring(bitstring_dropped=initial_state, num_qubits=num_qubits)
                    initial_state = np.array(list(initial_state), dtype=np.int).reshape((2*self.num_qubits, 2*self.num_qubits+1))
                    self.state = Clifford(initial_state)
                else:
                    raise ValueError("Initial state is a string with invalid length.")
            else:
                self.state = Clifford(initial_state)
        elif sampling_method == "random_walk":
            if high is None:
                high = int(20 * np.log(num_qubits) / np.log(2))
            seq_len = self.rng.integers(low=1, high=high)
            self.state = sequence_to_tableau(
                random_sequence(self.rng, seq_len, num_qubits), num_qubits
            )
        elif sampling_method == "uniform":
            self.state = random_clifford_uniform(num_qubits=num_qubits, seed=seed)
        else:
            raise NotImplementedError("sampling method not recognized")

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
            move_set.append((i, "h"))
            move_set.append((i, "s"))
            move_set.append((i, "sdg"))
            move_set.append((i, "x"))
            move_set.append((i, "y"))
            move_set.append((i, "z"))
            if i != self.num_qubits - 1:
                for j in range(i + 1, self.num_qubits):
                    move_set.append((i, j, "cx"))
                    move_set.append((j, i, "cx"))
                    move_set.append((i, j, "swap"))
                    # move_set.append((j, i, 'SWAP'))
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
            self.move_set[i]: sequence_to_tableau([self.move_set[i]], self.num_qubits)
            for i in range(len(self.move_set))
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
            move_arr = np.zeros((num_moves, 2 * self.num_qubits, 2 * self.num_qubits))
            for i_move in range(num_moves):
                move = self.move_set[i_move]
                move_arr[i_move, :, :] = 1 * self.move_set_tableau[move].tableau[:, :-1]
        else:
            move_arr = np.zeros(
                (num_moves, 2 * self.num_qubits, 2 * self.num_qubits + 1)
            )
            for i_move in range(num_moves):
                move = self.move_set[i_move]
                move_arr[i_move, :, :] = 1 * self.move_set_tableau[move].tableau
        return move_arr

    def apply_move(self, move: str, inplace: bool = False) -> Union[None, Clifford]:
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

    def random_move(self, inplace: bool = False) -> Tuple:
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
            if np.array_equal(
                1 * self.state.tableau[:, :-1], np.eye(2 * self.num_qubits)
            ):
                return True
        else:
            if self.state == sequence_to_tableau([], self.num_qubits):
                return True
        return False

    def find_move_from_a_to_b(
        self, tableau_a: Clifford, tableau_b: Clifford
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

    def to_bitstring(self, drop_phase_bits=None) -> str:
        """
        Flatten the tableau into a bitstring.
        Used to check whether a given tableau has been seen before.

        Returns
        -------
        str
            The bitstring.
        """
        if drop_phase_bits is None:
            drop_phase_bits = self.drop_phase_bits

        if not drop_phase_bits:
            return "".join(list(str(x) for x in 1 * self.state.tableau.flatten()))
        else:
            return "".join(list(str(x) for x in 1 * self.state.tableau[:,:-1].flatten()))

    
    def generate_neighbors(self) -> np.ndarray:
        """
        Generate the neighbors of the current state.
        Neighbors are formatted as numpy arrays.

        Returns
        -------
        np.ndarray
            An array of shape (M, 2*N, 2*N+1) or (M, 2*N, 2*N) containing the M neighbors 
            of the current state (the difference is whether the phase bits are dropped or not)

        """
        if self.drop_phase_bits:
            neighbors = (
                np.einsum(
                    "ij, mjk -> mik",
                    1 * self.state.tableau[:, :-1],
                    self.move_set_array,
                )
                % 2
            )
        else:
            """
            This is quite slow. I did some profiling experiments and confirmed that the bottleneck is
            the first line where the tableau composition is performed, and not the casting to a numpy
            array.

            To improve this, we would need to implement a vectorized version of the `_compose_general`
            classmethod defined here:
            https://qiskit.org/documentation/_modules/qiskit/quantum_info/operators/symplectic/clifford.html#Clifford.compose
            """
            neighbors = [
                self.state & tableau for tableau in self.move_set_tableau.values()
            ]
            neighbors = np.asarray([neighbor.tableau for neighbor in neighbors])

        return neighbors