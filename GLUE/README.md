# SPRA-LoRA and baseline algorithms with heterogeneous client ranks on GLUE benchmark

Federated fine-tuning of RoBERTa-base with LoRA family algorithms on GLUE, where clients use
different LoRA ranks (default `4,8,16,32,64,128`, assigned round-robin) and
data is split non-IID with a Dirichlet distribution.

## Methods

| Script | `--alg` |
|---|---|
| `main_LLM_hetrank.py` | `HetLoRA`, `RoLoRA`, `LA-LoRA`, `FedSA`, `FlexLoRA`, `FSLoRA` |
| `main_LLM_hetrank_stack.py` | `FFA`, `FLoRA`, `SPRA-LoRA` |

- **HetLoRA** averages zero-padded client *deltas* (changes since the
  round started). Rows a client did not train stay as they were.
- **RoLoRA** is HetLoRA, but trains only B on even rounds and only A on odd rounds.
- **LA-LoRA** is HetLoRA, but switches between B and A at every local step.
- **FedSA** shares A across clients. Each client keeps its own B. Its
  accuracy is the mean over that round's clients of (server A + own B).
- **SPRA-LoRA** with `--lr map` uses a learning rate per rank (`RANK_LR_MAP`).

Tasks (`--data_name`): `sst2`, `cola`, `RTE`, `MRPC`, `qnli`, `MNLI`, `QQP`.

## Setup

Python 3.10, CUDA 12.1.

```bash
conda create -n la_lora python=3.10 -y
conda activate la_lora
pip install -r requirements.txt
```


GLUE data and RoBERTa weights download from HuggingFace on first use, into
`~/.cache/huggingface`. To keep them somewhere else (e.g. a larger disk), set
`HF_HOME` before running:

```bash
export HF_HOME=/path/to/cache/huggingface
```

## Run one job

```bash
python main_LLM_hetrank.py --alg HetLoRA --data_name RTE \
    --num_clients 40 --client_frac 0.2 --rounds 150 --local_steps 20 \
    --ranks 4,8,16,32,64,128 --dirichlet_alpha 0.8 --lr 1e-4 \
    --lora_alpha 1 --batch_size 16 --seed 0 --output_dir runs/test
```

The stack script takes the same flags, plus `--baseline_eval`. That flag
scores the pretrained model with no training and logs it as round -1.

Each run writes a log (`<output_dir>/*.txt`, with lines like
`Round 5: acc=...`) and TensorBoard events to `--output_dir`.

## Plots

The plot scripts read runs laid out as
`runs/glue/<task>_n<clients>/<method>/<run name>/`. Each run folder needs an
`args.json` with the run's settings and a `.out` file holding the training
output, which ends with `Done.` when the run finished. Runs that aren't
finished are skipped.

```bash
# written to plots/glue/
python plot_scripts/plot_eval_curves.py --clients 40 60 80 --alphas 0.8
python plot_scripts/plot_bar_avg_across_datasets.py --clients 40
python plot_scripts/plot_bar_avg_across_clients.py
```

Result tables (mean ± standard error over seeds; add `--latex` or `--markdown`,
`--last` for last-round instead of best-round accuracy):

```bash
python plot_scripts/summarize.py --dir runs/glue --finished-only --n 40 --alpha 0.8
```

Method names, colors and display order for all plots are in
`plot_scripts/plot_utils.py`.

## Files

```
main_LLM_hetrank.py        HetLoRA-family, FlexLoRA, FSLoRA
main_LLM_hetrank_stack.py  FFA, FLoRA, SPRA-LoRA; --baseline_eval
dirichlet_data.py          non-IID client split
plot_scripts/              accuracy curves, bar charts, result tables (summarize.py)
```
