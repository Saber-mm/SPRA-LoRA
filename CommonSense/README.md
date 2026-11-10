# SPRA-LoRA and baseline algorithms with heterogeneous client ranks on Common Sense Reasoning dataset

Federated fine-tuning of LLaMA-3.2-3B with LoRA family algorithms on GLUE, where clients use
different LoRA ranks (default `4,8,16,32,64,128`, with uniform distribution) and
data is split non-IID with a Dirichlet distribution.

## Methods

| Script | `--alg` |
|---|---|
| `main_Stacked.py` | `FFA`, `FLoRA`, `SPRA-LoRA` |
| `main_HetLoRA.py` | `HetLoRA`, `RoLoRA`, `LA-LoRA` |
| `main_FSLoRA.py` | `FSloRA` |
| `main_FedSA.py` | `FedSA` |
| `main_FlexLoRA.py` | `FlexLoRA` |

- **HetLoRA** averages zero-padded client *deltas* (changes since the
  round started). Rows a client did not train stay as they were.
- **RoLoRA** is HetLoRA, but trains only B on even rounds and only A on odd rounds.
- **LA-LoRA** is HetLoRA, but switches between B and A at every local step.
- **FedSA** shares A across clients. Each client keeps its own B. Its
  accuracy is the mean over that round's clients of (server A + own B).
- **FSLoRA** uses a fixed global rank with sketching on clients side to update LoRA submatrices.
- **FLoRA** stacks clients LoRA components to aggregated client *deltas* of the adapter parameter.
- **FFA** (with extended aggregation strategy) stacks clients LoRA components (frozen *A* matrices) to aggregated client *deltas* in the adapter.
- **SPRA-LoRA** resamples and freezes LoRA *A* matrices, with `--lr map` uses a learning rate per rank (`RANK_LR_MAP`).

Tasks: `BoolQ`, `WinoGrande`, `OBQA`, `ARC-e`, `ARC-c`, `SIQA`, `PIQA`, `HellaSwag`

## Setup

Python 3.10, CUDA 12.1.

```bash
conda create -n commonsense_lora python=3.10 -y
conda activate commonsense_lora
pip install -r SPRA-LoRA/CommonSense_requirements.txt
```

For the commonsense reasoning benchmark, data is downloaded from **[Commonsense Reasoning Benchmark Dataset](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset)**. LLaMA-3.2-3B weights download from HuggingFace on first use into the repository directory.

## Run one job
The following are two examples for running experiments with HetLoRA and SPRA-LoRA:
```bash
python main_HetLoRA.py --algorithm=HetLoRA --dataset_seed=0
```
```bash
python main_Stacked.py --algorithm=SPRA-LoRA --dataset_seed=0
```
Important parameters are set in the arg.py file (or could be set when running jobs).

Each run saves the final model checkpoints in repository under a `./model_parameters_set/` directory.

## Evaluation

For evaluating a model fine-tuned using an algorithm, copy the path of the model saved in `./model_parameters_set/` into `saved_model_path` (line 92) of `evaluation_vllm.py` and run the python file.

## Plots

The plot scripts are in the `plots.ipynb` notebook. 
