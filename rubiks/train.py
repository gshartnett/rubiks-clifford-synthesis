import argparse
import os, sys
import pickle

import numpy as np
import torch

import rubiks.clifford as cl
import rubiks.lgf as lgf


parser = argparse.ArgumentParser(
    description="Train the Rubiks-inspired approach for Clifford synthesis."
)

parser.add_argument("--num_qubits", default=5, type=int, help="number of qubits")

parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="learning rate"
)

parser.add_argument("--batch_size", default=2000, type=int, help="batch size")

parser.add_argument("--num_epochs", default=1000, type=int, help="number of epochs")

parser.add_argument(
    "--eval_num_trials", default=2000, type=int, help="number of evaluation trials"
)

parser.add_argument(
    "--beam_width",
    default=5,
    type=int,
    help="Beam width for beam search",
)

parser.add_argument(
    "--eval_max_iter",
    default=1000,
    type=int,
    help="maximum number of iterations for evaluation",
)

parser.add_argument(
    "--scaling",
    default="linear",
    type=str,
    help="scaling of the high parameter with the number of qubits, must be one of log, linear, or log-linear",
)

parser.add_argument(
    "--sampling_method",
    default="random_walk",
    type=str,
    help="Sampling method used to generate Clifford elements.",
)

parser.add_argument(
    "--use_gpu",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Use the GPU, if available",
)

parser.add_argument(
    "--use_qiskit",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Boost the training by using the qiskit decomposition",
)

parser.add_argument(
    "--drop_phase_bits",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Drop the phase bits of the tableau",
)

parser.add_argument(
    "--load_saved_model",
    default=True,
    action=argparse.BooleanOptionalAction,
    help="Load a previously saved model, if available",
)

args = vars(parser.parse_args())

print("Run parameters\n==================================")
for key, value in args.items():
    print(f"{key}: {value}")

## if there is a GPU, use it
device = torch.device(
    "cuda" if (args["use_gpu"] and torch.cuda.is_available()) else "cpu"
)
print(f"\nRunning with device={device}\n")

## maximum number of moves to consider for random sequences
high = cl.max_random_sequence_length(args["num_qubits"], args["scaling"])
args["high"] = high  # add to dict for logging purposes
eval_max_iter = args["eval_max_iter"]
eval_num_trials = args["eval_num_trials"]
sampling_method = args["sampling_method"]
beam_width = args["beam_width"]

## save directory
n = args["num_qubits"]
scaling = args["scaling"]
if args["drop_phase_bits"]:
    data_dir = f"data/data_n_{n}_drop_phase_bits_scaling_{scaling}/"
else:
    data_dir = f"data/data_n_{n}_keep_phase_bits_scaling_{scaling}/"
eval_dir = data_dir + "eval_beamsearch/"
# timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data_dir = "data/" + timestamp
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

## run hyper-parameters
if not args["drop_phase_bits"]:
    WEIGHT_DICT = {
        "x": 1,
        "y": 1,
        "z": 1,
        "h": 1,
        "s": 1,
        "sdg": 1,
        "cx": 30,
        "swap": 90,
    }
else:
    WEIGHT_DICT = {
        "cx": 1,
        "swap": 3,
    }

use_qiskit = args["use_qiskit"]
SEED = 123

## iniitialize model
lgf_model = lgf.LGFModel(
    num_qubits=args["num_qubits"],
    device=device,
    rng=np.random.default_rng(SEED),
    hidden_layers=[32, 16, 4, 2],
    drop_phase_bits=args["drop_phase_bits"],
    use_qiskit=use_qiskit,
).to(device)

## either load previously trained model or train from scratch
if args["load_saved_model"] and os.path.exists(data_dir + "checkpoint"):
    lgf_model.load_state_dict(torch.load(data_dir + "checkpoint"))
else:
    ## train model
    print("training model:")
    loss_history = lgf_model.train(
        batch_size=args["batch_size"],
        learning_rate=args["learning_rate"],
        num_epochs=args["num_epochs"],
        weight_dict=WEIGHT_DICT,
        high=high,
    )

## save training results
if not (args["load_saved_model"] and os.path.exists(data_dir + "checkpoint")):
    torch.save(lgf_model.state_dict(), data_dir + "checkpoint")

    ## loss history
    np.save(data_dir + "loss_history", loss_history)

    ## command line args
    with open(data_dir + "args.pkl", "wb") as f:
        pickle.dump(args, f)

    ## gate weight dictionary
    with open(data_dir + "weight_dict.pkl", "wb") as f:
        pickle.dump(WEIGHT_DICT, f)

# check if file exists
steps_until_success_path = f"{eval_dir}steps_until_success_eval_max_iter_{eval_max_iter}_eval_num_trials_{eval_num_trials}_sampling_method_{sampling_method}.pkl"
if os.path.exists(steps_until_success_path):
    print(f"Evaluation file already exists, skipping!")
    sys.exit()

## evaluate steps until success
steps_until_success = {}

## lgf hillclimbing
print("evaluating hillclimbing using LGF")
steps_until_success["lgf"] = lgf.compute_weighted_steps_until_success(
    lgf_model=lgf_model,
    weight_dict=WEIGHT_DICT,
    num_trials=args["eval_num_trials"],
    max_iter=args["eval_max_iter"],
    method="hillclimbing",
    high=high,
    sampling_method=sampling_method,
)

## lgf beam search
print("evaluating beamsearch using LGF")
steps_until_success[f"beam-{beam_width}"] = lgf.compute_weighted_steps_until_success(
    lgf_model=lgf_model,
    weight_dict=WEIGHT_DICT,
    num_trials=args["eval_num_trials"],
    max_iter=args["eval_max_iter"],
    method="beamsearch",
    high=high,
    sampling_method=sampling_method,
    beam_width=beam_width,
)

## qiskit method
print("evaluating qiskit")
steps_until_success["qiskit"] = lgf.compute_weighted_steps_until_success(
    lgf_model=lgf_model,
    weight_dict=WEIGHT_DICT,
    num_trials=args["eval_num_trials"],
    max_iter=args["eval_max_iter"],
    method="qiskit",
    high=high,
    sampling_method=sampling_method,
)

## save evaluation results
eval_max_iter = args["eval_max_iter"]
with open(steps_until_success_path, "wb") as f:
    pickle.dump(steps_until_success, f)

## print out some statistics
bitstring_qiskit = [
    bitstr
    for bitstr in steps_until_success["qiskit"]["bitstrings"]
    if bitstr != None
]
bitstring_lgf = [
    bitstr for bitstr in steps_until_success["lgf"]["bitstrings"] if bitstr != None
]
print(
    f"(Qiskit) unique tableaus encountered: {len(set(bitstring_qiskit))}, total tableaus encountered: {len(bitstring_qiskit)}"
)
print(
    f"(LGF) unique tableaus encountered: {len(set(bitstring_lgf))}, total tableaus encountered: {len(bitstring_lgf)}\n"
)
