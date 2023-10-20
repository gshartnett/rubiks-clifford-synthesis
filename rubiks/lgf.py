import copy
from datetime import datetime
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
import torch
import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full
from torch import (
    nn,
    optim,
)
from torch.utils.tensorboard import SummaryWriter
import time
import hashlib
import rubiks.clifford as cl


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
        drop_phase_bits=False,
        use_qiskit=False,
    ):
        super(LGFModel, self).__init__()
        self.num_qubits = num_qubits
        self.drop_phase_bits = drop_phase_bits
        self.use_qiskit = use_qiskit  # can remove support

        if drop_phase_bits:
            self.input_dim = 2 * num_qubits * (2 * num_qubits)
        else:
            self.input_dim = 2 * num_qubits * (2 * num_qubits + 1)

        self.hidden_layers = [self.input_dim] + hidden_layers + [1]
        self.device = device
        self.rng = rng
        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1])
                for i in range(len(self.hidden_layers) - 1)
            ]
        )
        ## nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d_1 = nn.Conv2d(1, 1, 2, stride=1)
        # self.conv2d_2 = nn.Conv2d(1, 1, 2, stride=1) # can either use or remove support

        # H_input = 2*self.num_qubits + 1
        # W_input = 2*self.num_qubits
        # H_output, W_output = conv2d_output_dimensions(H_input, W_input, 2, 1)
        # H_output, W_output = conv2d_output_dimensions(H_output, W_output, 2, 1)
        # self.fc1 = nn.Linear(H_output * W_output, 1)

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
            x_tensor = torch.tensor(
                1.0 * x.tableau.flatten()[None, :-1], dtype=torch.float32
            )
        else:
            x_tensor = torch.tensor(
                1.0 * x.tableau.flatten()[None, :], dtype=torch.float32
            )
        with torch.no_grad():
            return self.forward(x_tensor).item()

    def train(
        self,
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        weight_dict: Dict,
        high: Union[None, int] = None,
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
        writer = SummaryWriter(f"runs/date_{date}_num_qubits_{self.num_qubits}")
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100,)
        loss_current = np.inf

        ## loop over "epochs"
        pbar = tqdm.trange(num_epochs)
        for epoch in pbar:
            ## update the progress bar
            pbar.set_description(f"loss={loss_current:.4f}, lr={learning_rate:.3e}")

            ## sample a batch of Cliffords
            x_batch, seq_lens = cl.generate_data_batch(
                self.rng,
                batch_size,
                self.num_qubits,
                self.device,
                weight_dict,
                high,
                self.drop_phase_bits,
                self.use_qiskit,
            )
            out = self.forward(x_batch)

            # loss = torch.nn.MSELoss(reduction='mean')(out, seq_lens)
            # loss = torch.mean(torch.square(out - seq_lens))
            loss = -pearson_correlation(out, seq_lens)

            writer.add_scalar("loss", loss, epoch)
            writer.add_scalar("lr", learning_rate, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            learning_rate = scheduler.get_last_lr()[0]

            # loss_current = loss.detach().numpy().item()
            loss_current = loss.item()
            _loss_history.append(loss_current)

        writer.close()
        return _loss_history


def pearson_correlation(input_sequence_1: torch.Tensor, input_sequence_2: torch.Tensor):
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
    pearson_eps = 1e-10  # used to regulate correlation calculation

    delta_1 = input_sequence_1 - torch.mean(input_sequence_1, axis=0, keepdim=True)
    delta_2 = input_sequence_2 - torch.mean(input_sequence_2, axis=0, keepdim=True)
    corr = torch.sum(delta_1 * delta_2, dim=0, keepdim=True)
    corr = corr / (
        torch.sqrt(torch.sum(delta_1 ** 2 + pearson_eps, dim=0, keepdim=True))
        * torch.sqrt(torch.sum(delta_2 ** 2 + pearson_eps, dim=0, keepdim=True))
    )
    corr = torch.mean(corr)
    return corr


def conv2d_output_dimensions(
    height_input: int,
    width_input: int,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    padding: int = 0,
) -> Tuple[int, int]:
    """
    Helper function used to determine how the shape of a tensor changes after
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
    height_output = (height_input + 2 * padding - dilation * (kernel_size - 1) - 1) / (
        stride
    ) + 1
    width_output = (width_input + 2 * padding - dilation * (kernel_size - 1) - 1) / (
        stride
    ) + 1
    return int(height_output), int(width_output)


def hillclimbing_random(
    initial_state: Clifford, max_iter: int = 1000, seed: int = 123,
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
    problem = cl.Problem(
        initial_state.tableau.shape[0] // 2,
        initial_state=initial_state.copy(),
        seed=seed,
    )
    identity_tableau = cl.sequence_to_tableau([], problem.num_qubits).tableau
    result = {"success": False, "move_history": []}

    for _ in range(max_iter):
        ## check for solution
        if np.array_equal(problem.state.tableau, identity_tableau):
            result["success"] = True
            return result

        ## make a random move
        move, _ = problem.random_move()
        result["move_history"].append(move)
        problem.apply_move(move, inplace=True)

    return result


def hillclimbing(
    initial_state: Clifford,
    lgf_model: nn.Module,
    max_iter: int = 1000,
    seed: int = 123,
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
    assert initial_state.tableau.shape[0] // 2 == lgf_model.num_qubits
    result = {"success": False, "move_history": [], "L_history": [], "move_type": []}

    ## for now, do evaluation on CPU
    lgf_model = lgf_model.to("cpu")

    ## initialize
    problem = cl.Problem(
        lgf_model.num_qubits,
        initial_state=initial_state.copy(),
        seed=seed,
        drop_phase_bits=lgf_model.drop_phase_bits,
    )
    num_moves = len(problem.move_set)
    lgf_value = np.inf

    ## build the identity tableau (as a numpy array)
    if problem.drop_phase_bits:
        identity = np.eye(2 * problem.num_qubits)
    else:
        identity = 1 * Clifford(QuantumCircuit(lgf_model.num_qubits)).tableau

    ## check for solution
    if problem.is_solution():
        result["success"] = True
        return result

    ## loop over iterations
    for _ in range(max_iter):

        ## build array of candidates
        candidates = problem.generate_neighbors()

        ## check to see if any of the candidates are the solution
        for i_move in range(num_moves):
            if problem.drop_phase_bits:
                ## confirm the array shape is correct
                assert candidates.shape == (
                    len(problem.move_set),
                    2 * problem.num_qubits,
                    2 * problem.num_qubits,
                )

                if np.array_equal(candidates[i_move], identity):
                    result["move_history"].append(problem.move_set[i_move])
                    result["move_type"].append("Solution")
                    result["success"] = True
                    with torch.no_grad():
                        result["L_history"].append(
                            lgf_model.forward(
                                torch.tensor(
                                    candidates[i_move].flatten()[None, :],
                                    dtype=torch.float32,
                                )
                            )
                        )
                    return result
            else:
                ## confirm the array shape is correct
                assert candidates.shape == (
                    len(problem.move_set),
                    2 * problem.num_qubits,
                    2 * problem.num_qubits + 1,
                )

                if np.array_equal(candidates[i_move], identity):
                    result["move_history"].append(problem.move_set[i_move])
                    result["move_type"].append("Solution")
                    result["success"] = True
                    with torch.no_grad():
                        result["L_history"].append(
                            lgf_model.forward(
                                torch.tensor(
                                    candidates[i_move].flatten()[None, :],
                                    dtype=torch.float32,
                                )
                            )
                        )
                    return result

        ## evaluate LGF for each candidate
        candidates = torch.tensor(candidates, dtype=torch.float32)
        candidates = torch.flatten(candidates, start_dim=1)
        with torch.no_grad():
            lgf_of_candidates = lgf_model.forward(candidates).numpy()

        ## pick the best candidate (with random tie breaking)
        # i_best = np.argmin(lgf_of_candidates)
        i_best = np.random.choice(
            np.flatnonzero(np.isclose(lgf_of_candidates, lgf_of_candidates.min()))
        )
        lgf_best = lgf_of_candidates[i_best]

        ## apply move (best or random)
        if lgf_best < lgf_value:
            lgf_value = lgf_best
            result["L_history"].append(lgf_value)
            result["move_history"].append(problem.move_set[i_best])
            result["move_type"].append("LGF")
            problem.apply_move(problem.move_set[i_best], inplace=True)
        else:
            i_move = lgf_model.rng.choice(len(problem.move_set))
            move = problem.move_set[i_move]
            lgf_value = lgf_of_candidates[i_move]
            result["L_history"].append(lgf_value)
            result["move_history"].append(move)
            result["move_type"].append("Random")
            problem.apply_move(move, inplace=True)

    return result


def beam_search(
    initial_state: Clifford,
    lgf_model: nn.Module,
    beam_width: int = 1,
    max_iter: int = 100,
    drop_phase_bits=True,
    check = False,
) -> Dict:
    """
    1. batch the computation of the LGF
    2. only compute LL for retained guys

    Beam search function using the learned guidance function output as the heuristic.
    If the search succeeds, then it will produce a linked list whose node sequence implements the inverse of the initial Clifford.
    In particular, 
        initial_state @ cl.sequence_to_tableau(results_dict['linked_list'].get_move_list()[:-1][::-1], num_qubits=num_qubits)
    will be the identity. The [:-1] removes the initial move (None), and the [::-1] reverses the sequence since the linked list
    is ordered from most recently visited node to the starting node (i.e., in reverse order).

    For beam_width=1 reduces to the Greedy "hillclimbing" algorithm.

    Parameters
    ----------
    initial_state : Clifford
        The initial state.
    lgf_model : nn.Module
        A learned guidance function model.
    beam_width : int, optional
        The beam width, by default 1.
    max_iter : int, optional
        The maximum number of steps to consider, by default 100.
    drop_phase_bits : bool, optional
        Whether the phase bits should be dropped, by default True.

    Returns
    -------
    Dict
        A results dictionary.
    """
    # initialize the starting node
    problem = cl.Problem(
        lgf_model.num_qubits, 
        initial_state=initial_state, 
        drop_phase_bits=drop_phase_bits
        )
    num_qubits = problem.num_qubits
    num_moves = len(problem.move_set)
    
    # check for solution
    if problem.is_solution():
        return {
            "success": True,
            "count": 0,
            "linked_list": None,
            "move_history": []
            }

    # initialize the linked list
    linked_list = cl.LinkedList()
    linked_list.insertAtBegin(data={"state": problem.to_bitstring(drop_phase_bits=drop_phase_bits), "move": None})

    # identity tensor
    identity_tensor = 1 * cl.sequence_to_tableau([], num_qubits).tableau
    if drop_phase_bits:
        identity_tensor = identity_tensor[:, :-1]

    # perform the search
    beam = [linked_list]
    visited_nodes = {problem.to_bitstring()}
    count = 1
    while (len(beam) > 0) and (count < max_iter):

        # loop over nodes in current beam (maximum of beam_width items)
        # build a list of M neighbors for each beam node
        time_point = time.time()
        neighbor_tensors = None
        for ll in beam:

            # grab the neighbors
            node = ll.head
            node_state = node.data["state"]
            node_problem = cl.Problem(
                num_qubits=num_qubits, 
                initial_state=node_state, 
                drop_phase_bits=drop_phase_bits,
            )

            # stack the neighbor tableaus into one big array
            if neighbor_tensors is None:
                neighbor_tensors = node_problem.generate_neighbors()  # (M, 2N, 2N) array
            else:
                neighbor_tensors = np.concatenate((neighbor_tensors, node_problem.generate_neighbors())) # (num_moves * M, 2N, 2N) array
        #print(f"Time taken to build neighbor list of beams: {time.time() - time_point}")
        #time_point = time.time()

        # list of the parent linked_list indices and move index for each neighbor
        parent_list = [(i // num_moves, i % num_moves) for i in range(neighbor_tensors.shape[0])] 

        # check for solution
        for i_neighbor, (idx_beam, idx_move) in enumerate(parent_list):

            #assert i_neighbor == idx_beam * num_moves + idx_move
            
            neighbor_tensor = neighbor_tensors[i_neighbor]

            if np.array_equal(neighbor_tensor, identity_tensor):

                # pad state with a trivial set of phase bits
                if drop_phase_bits:
                    neighbor_tensor = np.hstack((neighbor_tensor, np.zeros(2*num_qubits)[:, None]))

                neighbor_problem = cl.Problem(
                    num_qubits=lgf_model.num_qubits,
                    initial_state=Clifford(neighbor_tensor),
                    drop_phase_bits=drop_phase_bits,
                )
                neighbor_state = neighbor_problem.to_bitstring(drop_phase_bits=drop_phase_bits)
                move = problem.move_set[idx_move]
                ll_new = beam[idx_beam].copy()
                ll_new.insertAtBegin(data={"state": neighbor_state, "move": move})
                move_history = ll_new.get_move_list()[:-1][::-1]

                if check:
                    state = initial_state @ cl.sequence_to_tableau(move_history, num_qubits=num_qubits)
                    if not drop_phase_bits:
                        if not np.array_equal(state.tableau, cl.sequence_to_tableau([], num_qubits=num_qubits).tableau):
                            raise ValueError("Error, solution identity is not respected, likely problem with beam_search function.")
                    else:
                        if not np.array_equal(state.tableau[:,:-1], cl.sequence_to_tableau([], num_qubits=num_qubits).tableau[:,:-1]):
                            raise ValueError("Error, solution identity is not respected, likely problem with beam_search function.")

                return {
                    "success": True,
                    "count": count + 1,
                    "linked_list": ll_new,
                    "move_history": move_history
                }
        #print(f"Time taken to check for solution: {time.time() - time_point}")
        time_point = time.time()

        # compute LGF values for neighbors
        neighbor_tensors = torch.tensor(neighbor_tensors, dtype=torch.float32)
        with torch.no_grad():
            lgf_of_neighbors = lgf_model.forward(torch.flatten(neighbor_tensors, start_dim=1))
        #print(f"Time taken to compute LGF values: {time.time() - time_point}")
        time_point = time.time()

        # update beam
        new_beam = []
        lgf_of_neighbors_indices_sorted = torch.argsort(lgf_of_neighbors)
        neighbor_count = 0
        #print(f"Number of neighbors to sort through: {len(lgf_of_neighbors_indices_sorted)}")
        while (len(new_beam) < beam_width) and (neighbor_count < len(lgf_of_neighbors_indices_sorted)):
            i_neighbor = lgf_of_neighbors_indices_sorted[neighbor_count]
            idx_beam, idx_move = parent_list[i_neighbor]
            neighbor_tensor = neighbor_tensors[i_neighbor].numpy()
            #if drop_phase_bits:
            #    neighbor_tensor = np.hstack((neighbor_tensor, np.zeros(2*num_qubits)[:, None]))
            #neighbor_problem = cl.Problem(
            #    num_qubits=lgf_model.num_qubits,
            #    initial_state=Clifford(neighbor_tensor),
            #    drop_phase_bits=drop_phase_bits,
            #)
            #neighbor_bitstring = neighbor_problem.to_bitstring(drop_phase_bits=drop_phase_bits)
            #print(len(neighbor_bitstring))
            #print(neighbor_bitstring)
            neighbor_bitstring = "".join(list(str(int(x)) for x in 1 * neighbor_tensor.flatten()))            
            #print(len(neighbor_bitstring))
            #print(neighbor_bitstring)
            #assert 1==0

            if neighbor_bitstring not in visited_nodes:
            #if True:
                #print(f"neighbor_count={neighbor_count}")
                ll_new = beam[idx_beam].copy()
                
                #x = ll_new.head.data['state']
                #old_beam_state = hashlib.sha1(x.encode("utf-8")).hexdigest()

                move = problem.move_set[idx_move]
                ll_new.insertAtBegin(data={"state": neighbor_bitstring, "move": move})
                new_beam.append(ll_new)

                #if neighbor_bitstring in visited_nodes:
                    #x = ll_new.head.data['state']
                    #new_beam_state = hashlib.sha1(x.encode("utf-8")).hexdigest()
                    #print('\n I"VE BEEN HERE BEFORE!"')
                    #print('move =', move)
                    #print('old beam state = ', old_beam_state)
                    #print('new beam state = ', new_beam_state)

                visited_nodes.add(neighbor_bitstring)

            neighbor_count += 1

        beam = new_beam
        count += 1
        #print(f"Time taken to update beam: {time.time() - time_point}")
        time_point = time.time()

    return {
        "success": False,
        "count": count,
        "linked_list": None,
        "move_history": None,
    }


def compute_weighted_steps_until_success(
    lgf_model: nn.Module,
    weight_dict: Dict,
    num_trials: int = 100,
    max_iter: int = int(1e4),
    method: str = "hillclimbing",
    high: Union[int, None] = None,
    sampling_method="random_walk",
    beam_width: int = 5,
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
        Method to use, by default 'hillclimbing'.

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
    bitstrings = []
    steps_until_success = []
    CNOT_count = []
    for k in tqdm.trange(num_trials):
        ## instantiate each problem with a new seed not used in the training
        problem = cl.Problem(
            lgf_model.num_qubits,
            high=high,
            seed=2305843009213693951 + k,
            sampling_method=sampling_method,
        )

        ## apply the synthesis method
        if method == "hillclimbing":
            # I don't think the seed is being used here
            result = hillclimbing(problem.state, lgf_model, max_iter=max_iter, seed=k,)
        elif method == "beamsearch":
            result = beam_search(
                initial_state=problem.state,
                lgf_model=lgf_model,
                beam_width=beam_width,
                max_iter=max_iter,
            )
        elif method == "random":
            result = hillclimbing_random(problem.state, max_iter=max_iter, seed=k,)
        elif method == "qiskit":
            ## hacky way to put the Qiskit results in the same data structure as the above
            result = {
                "success": True,
                "move_history": synth_clifford_full(problem.state),
            }
        else:
            raise NotImplementedError("synthesis method not recognized")

        ## compute the weighted distance and CNOT count
        if result["success"]:
            bitstrings.append(problem.to_bitstring())
            steps_until_success.append(
                cl.weighted_distance_to_identity(result["move_history"], weight_dict)
            )
            CNOT_count.append(
                cl.weighted_distance_to_identity(
                    result["move_history"], weight_dict, CNOTs_only=True
                )
            )
        else:
            bitstrings.append(None)
            steps_until_success.append(None)
            CNOT_count.append(None)

    return {
        "bitstrings": bitstrings,
        "weighted_steps": steps_until_success,
        "CNOTs": CNOT_count,
    }
