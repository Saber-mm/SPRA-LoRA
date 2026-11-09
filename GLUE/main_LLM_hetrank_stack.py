import os, math, time, random, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import load_dataset
from transformers import (
    RobertaForSequenceClassification, RobertaTokenizer,
    AutoModelForSequenceClassification,
)
from peft import LoraConfig, PeftModel, TaskType
from tensorboardX import SummaryWriter
import argparse

from dirichlet_data import data_from_dirichlet

# ── argument parsing ───────────────────────────────────────────────────────────

def str2bool(v):
    return str(v).lower() == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--alg',         default='FLoRA',   type=str,
                    choices=['FLoRA', 'FFA', 'Frank-LoRA'])
parser.add_argument('--data_name',   default='sst2',    type=str)
parser.add_argument('--num_clients', default=20,        type=int)
parser.add_argument('--client_frac', default=0.2,       type=float, help='fraction of clients per round')
parser.add_argument('--rounds',      default=150,       type=int,   help='communication rounds')
parser.add_argument('--local_steps', default=20,        type=int,   help='local steps per round')
parser.add_argument('--batch_size',  default=16,        type=int)
parser.add_argument('--lr',          default=5e-4,
                    type=lambda x: None if str(x).lower() == 'map' else float(x),
                    help='client learning rate; pass "map" to use RANK_LR_MAP')
parser.add_argument('--optimizer',   default='adamw',   type=str, choices=['adamw', 'sgd'],
                    help='client optimizer (adamw or sgd)')
parser.add_argument('--momentum',    default=0.9,       type=float, help='momentum for sgd optimizer')
parser.add_argument('--ranks',       default='1,2,4,8', type=str,
                    help='comma-separated client rank values, distributed round-robin')
parser.add_argument('--dirichlet_alpha', default=0.1,       type=float, help='Dirichlet alpha')
parser.add_argument('--lora_alpha',  default=16,        type=int)
parser.add_argument('--lora_dropout',default=0.1,       type=float)
parser.add_argument('--gpu',         default='0',       type=str)
parser.add_argument('--eval_every',  default=5,         type=int,   help='eval every N rounds')
parser.add_argument('--output_dir',  required=True,     type=str,   help='directory for logs, TensorBoard events and the merged model')
parser.add_argument('--model',       default='roberta_base', type=str,
                    choices=['roberta_base'])
parser.add_argument('--seed',        default=42,        type=int)
parser.add_argument('--lora_scaling', default=0.0,        type=float,
                    help='if non-zero, set client lora_alpha = rank * lora_scaling')
parser.add_argument('--baseline_eval', action='store_true',
                    help='evaluate the frozen pretrained model and exit (no training); logs at step -1')
parser.add_argument('--freeze_classifier', type=str2bool, default=True,
                    help='freeze the classifier head full-rank weights (classifier.dense.base_layer '
                         'and classifier.out_proj). LoRA on classifier.dense remains trainable. '
                         'Pass "false" to disable.')
args = parser.parse_args()


def make_client_optimizer(params, lr):
    """Build the client optimizer selected by --optimizer (adamw or sgd), with default weight decay."""
    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=args.momentum)
    return torch.optim.AdamW(params, lr=lr)

# ── per-rank learning rate map ─────────────────────────────────────────────────
# Used when --lr map is passed. Maps client rank → learning rate.
# lr(r) = 2e-4 * sqrt(r / 4), from the CoLA AdamW homogeneous-rank sweep.
RANK_LR_MAP: dict[int, float] = {
    4:   2e-4,
    8:   2.83e-4,
    16:  4e-4,
    32:  5.66e-4,
    64:  8e-4,
    128: 1.13e-3,
    256: 1.6e-3,
}

clients_per_round = max(1, int(args.num_clients * args.client_frac))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── model / tokenizer ──────────────────────────────────────────────────────────

if args.model == 'roberta_base':
    MODEL_NAME    = 'FacebookAI/roberta-base'
    tokenizer     = RobertaTokenizer.from_pretrained(MODEL_NAME)
    TARGET_MODULES = ["query", "value", "key", "dense"]
    _TASK_TYPE    = TaskType.SEQ_CLS


_HF_CACHE = os.environ.get("HF_CACHE_DIR", None)


def build_base_model(num_labels=2):
    return RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
        attn_implementation="sdpa",
        cache_dir=_HF_CACHE,
    )


def build_model_from_checkpoint(path):
    """Load the merged model saved at the end of the previous round."""
    return AutoModelForSequenceClassification.from_pretrained(
        path,
        attn_implementation="sdpa",
        cache_dir=_HF_CACHE,
    )


def add_adapter(model, r, lora_alpha=None, target_modules=None):
    """Wrap model (in place, no copy) with a rank-r LoRA adapter."""
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha or args.lora_alpha,
        target_modules=target_modules or TARGET_MODULES,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=_TASK_TYPE,
        init_lora_weights="gaussian",
        use_rslora=True,
    )
    peft_model = PeftModel(model, cfg)
    if args.freeze_classifier:
        # Freeze classifier head's full-rank weights (modules_to_save copies),
        # but keep LoRA params on classifier.dense trainable.
        for n, p in peft_model.named_parameters():
            if "classifier" in n and "lora_" not in n and p.requires_grad:
                p.requires_grad = False
    return peft_model


# ── dataset loading ────────────────────────────────────────────────────────────

GLUE_CONFIGS = {
    'sst2':  dict(task='sst2',  text1='sentence',    text2=None,         num_labels=2, val_split='validation'),
    'cola':  dict(task='cola',  text1='sentence',    text2=None,         num_labels=2, val_split='validation'),
    'mrpc':  dict(task='mrpc',  text1='sentence1',   text2='sentence2',  num_labels=2, val_split='validation'),
    'rte':   dict(task='rte',   text1='sentence1',   text2='sentence2',  num_labels=2, val_split='validation'),
    'qnli':  dict(task='qnli',  text1='question',    text2='sentence',   num_labels=2, val_split='validation'),
    'qqp':   dict(task='qqp',   text1='question1',   text2='question2',  num_labels=2, val_split='validation'),
    'mnli':  dict(task='mnli',  text1='premise',     text2='hypothesis', num_labels=3, val_split='validation_matched'),
}


_glue_cfg = GLUE_CONFIGS[args.data_name.lower()]

def _preprocess(examples):
    if _glue_cfg['text2']:
        return tokenizer(examples[_glue_cfg['text1']], examples[_glue_cfg['text2']],
                         truncation=True, padding='max_length', max_length=128)
    return tokenizer(examples[_glue_cfg['text1']],
                     truncation=True, padding='max_length', max_length=128)

_raw      = load_dataset("glue", _glue_cfg['task'])
tokenized = _raw.map(_preprocess, batched=True, num_proc=4)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataset = tokenized['train']
test_dataset  = tokenized[_glue_cfg['val_split']]
num_labels    = _glue_cfg['num_labels']

# ── data partitioning ──────────────────────────────────────────────────────────
_nums_sample = len(train_dataset) // args.num_clients
data_idx, _  = data_from_dirichlet(args.data_name, args.dirichlet_alpha,
                                    num_labels, args.num_clients, _nums_sample)


def get_loader(pid):
    return DataLoader(train_dataset, batch_size=args.batch_size,
                      sampler=SubsetRandomSampler(data_idx[pid]), num_workers=0)


def get_test_loader():
    return DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ── evaluation ─────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()

# Extra info from the most recent evaluate() call on a binary GLUE task.
# CoLA accuracy is degenerate (69.13% = always-majority), so the predicted
# class count shows whether the model is just a constant predictor.
LAST_EVAL: dict = {}


def evaluate(model, loader):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    pred1 = 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            tgt    = batch['label'].to(device)
            logits = model(ids, attention_mask=mask).logits
            pred   = logits.argmax(dim=-1)
            correct += (pred == tgt).sum().item()
            total   += tgt.size(0)
            pred1   += (pred == 1).sum().item()
    model.to('cpu')
    LAST_EVAL.clear()
    if total > 0 and logits.size(-1) == 2:
        LAST_EVAL['pred1'] = pred1   # how many test examples got class 1
        LAST_EVAL['n'] = total
    return 100.0 * correct / total

# ── logging ────────────────────────────────────────────────────────────────────

_out_dir = Path(args.output_dir)
_out_dir.mkdir(parents=True, exist_ok=True)
log_name = (f"{args.alg}-{args.data_name}-{args.model}-{args.lr}"
            f"-{args.num_clients}-{args.batch_size}-{args.local_steps}"
            f"-{args.ranks.replace(',', '_')}")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(_out_dir / f"{log_name}.txt")
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)
writer = SummaryWriter(log_dir=str(_out_dir))

# ── rank assignment ────────────────────────────────────────────────────────────

def assign_ranks(N, ranks):
    """Distribute N clients uniformly (round-robin) across the given rank values."""
    return [ranks[i % len(ranks)] for i in range(N)]

rank_values  = [int(r) for r in args.ranks.split(',')]
client_ranks = assign_ranks(args.num_clients, rank_values)


def client_lora_alpha(r):
    """Return lora_alpha for an adapter of rank r. Uses lora_scaling if specified."""
    if args.lora_scaling:
        return int(r * args.lora_scaling)
    return args.lora_alpha

# ── helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_model_memory(model):
    model.train()
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.gradient_checkpointing_enable()
    return model

# ── stacking (FLoRA / FFA / Frank-LoRA) ───────────────────────────────────────

def train_stacking(frozen_model, client_dataloaders):
    if args.baseline_eval:
        baseline_acc = evaluate(frozen_model, get_test_loader())
        writer.add_scalar('accuracy', baseline_acc, -1)
        logger.info(f"Round -1 (baseline): acc={baseline_acc:.2f}  "
                    f"alg={args.alg}  data={args.data_name}  ranks={args.ranks}")
        print(f"[{args.alg}] Round -1 (baseline): acc={baseline_acc:.2f}", flush=True)
        return

    # Draw every round's clients up front from --seed. The per-client set_seed()
    # calls below reset the global RNG so add_adapter samples the intended A; drawing
    # clients from that RNG would tie each round's clients to the previous round's.
    client_gen = torch.Generator().manual_seed(args.seed)
    client_draws = [torch.randperm(len(client_ranks), generator=client_gen)[:clients_per_round]
                    for _ in range(args.rounds)]

    best_acc = 0.0

    pbar = tqdm(range(args.rounds))
    for rnd in pbar:
        client_ids = client_draws[rnd]
        selected_sum = sum(client_ranks[i] for i in client_ids)
        # updated merged "frozen_model":
        server_model = add_adapter(frozen_model, selected_sum, lora_alpha=client_lora_alpha(selected_sum))
        server_model.train()
        server_lora_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}
        if not server_lora_params:
            raise ValueError("no trainable parameters in server model")
        # Non-LoRA trainable params (e.g. an unfrozen classifier) are rebuilt as the
        # average of client values below, so start them from zero, not the old weights.
        with torch.no_grad():
            for n, p in server_lora_params.items():
                if "lora_" not in n:
                    p.zero_()

        pre_step_losses = []
        post_step_losses = []
        stack_index = 0
        for i, client_id in enumerate(client_ids):
            if args.alg == 'FFA':
                set_seed(args.seed) # sampling the same (submatrices of) A for all client at each round
            else: # i.e. ['FLoRA', 'Frank-LoRA']
                set_seed(rnd) # the same (submatrices of) A for all clients at each round, but different across rounds
            # starting from the updated frozen_model (merging):
            r_c = client_ranks[client_id]
            client_model = add_adapter(deepcopy(frozen_model), r_c, lora_alpha=client_lora_alpha(r_c))
            client_lora_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad}
            if args.alg in ['FFA', 'Frank-LoRA']:
                for n, p in client_model.named_parameters():
                    if "lora_A" in n:
                        p.requires_grad = False

            client_model = optimize_model_memory(client_model)
            client_lr = RANK_LR_MAP[r_c] if args.lr is None else args.lr
            client_opt = make_client_optimizer(client_model.parameters(), lr=client_lr)
            client_opt.zero_grad()
            client_loader = client_dataloaders[client_id]
            client_model = client_model.to(device)

            train_data_list = list(client_loader)
            first_batch = None
            step_trace = []  # per-step training loss for the first client (i==0)
            for step in range(args.local_steps):
                batch = random.choice(train_data_list)
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)
                outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if step == 0:
                    pre_step_losses.append(loss.item())
                    first_batch = (input_ids, attention_mask, labels)
                if i == 0:
                    step_trace.append(loss.item())
                loss.backward()
                client_opt.step()
                client_opt.zero_grad()

            # Per-step local-training loss curve for the first client of the round.
            if i == 0 and step_trace:
                trace_str = " ".join(f"{v:.3f}" for v in step_trace)
                print(f"[{args.alg}] Round {rnd}: client0 step-loss: {trace_str}", flush=True)
                logger.info(f"Round {rnd}: client0 step-loss: {trace_str}")

            # Post-finetuning loss on the SAME first batch (fair pre/post comparison).
            if first_batch is not None:
                client_model.eval()
                with torch.no_grad():
                    fi, fa, fl = first_batch
                    post_out = client_model(input_ids=fi, attention_mask=fa, labels=fl)
                    post_step_losses.append(post_out.loss.item())
                client_model.train()

            client_model = client_model.to('cpu')

            # Aggregating local adapters by stacking:
            s_server = 1 / math.sqrt(selected_sum) # scaling factor
            s_c = 1 / math.sqrt(r_c) # scaling factor for client
            for n in server_lora_params.keys():
                if "lora_B" in n:
                    server_lora_params[n].data[:, stack_index:(stack_index+r_c)] = client_lora_params[n].data.detach()
                elif "lora_A" in n:
                    server_lora_params[n].data[stack_index:(stack_index+r_c), :] = client_lora_params[n].data.detach() * (s_c / s_server) / clients_per_round
                else:
                    server_lora_params[n].data += client_lora_params[n].data.detach() * (1 / clients_per_round)

            stack_index += r_c
            if stack_index==selected_sum and i < clients_per_round-1:
                raise ValueError(f"LoRA stacking mismatch: got {stack_index}, expected {selected_sum}")

            # Clean up after client training
            del client_model
            torch.cuda.empty_cache()

        avg_pre_loss = sum(pre_step_losses) / len(pre_step_losses)
        avg_post_loss = sum(post_step_losses) / len(post_step_losses) if post_step_losses else float('nan')
        delta = avg_pre_loss - avg_post_loss
        print(f"[{args.alg}] Round {rnd}: pre-step loss={avg_pre_loss:.4f} post-step loss={avg_post_loss:.4f} (delta={delta:+.4f})", flush=True)
        logger.info(f"Round {rnd}: pre-step loss={avg_pre_loss:.4f} post-step loss={avg_post_loss:.4f} (delta={delta:+.4f})")

        save_path = str(_out_dir)
        os.makedirs(save_path, exist_ok=True)
        raw_model = server_model
        merged_model = raw_model.merge_and_unload()
        merged_model.eval()
        merged_model.save_pretrained(save_path)
        del merged_model, raw_model
        del server_model

        torch.cuda.empty_cache()
        frozen_model = build_model_from_checkpoint(save_path) # loading for the next round

        if rnd % args.eval_every == 0:
            acc      = evaluate(frozen_model, get_test_loader())
            best_acc = max(best_acc, acc)
            writer.add_scalar('accuracy', acc, rnd)
            extra = ""
            if 'pred1' in LAST_EVAL:
                extra = f"  pred1={LAST_EVAL['pred1']}/{LAST_EVAL['n']}"
            logger.info(f"Round {rnd}: acc={acc:.2f}  best={best_acc:.2f}{extra}  "
                        f"alg={args.alg}  data={args.data_name}  ranks={args.ranks}")
            print(f"[{args.alg}] Round {rnd}: acc={acc:.2f}  best={best_acc:.2f}{extra}")

# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(args.seed)

    if args.lr is None:
        print(f"RANK_LR_MAP (active): {RANK_LR_MAP}", flush=True)
    logger.info(
        f"alg={args.alg} data={args.data_name} model={args.model} "
        f"num_clients={args.num_clients} clients_per_round={clients_per_round} "
        f"local_steps={args.local_steps} rounds={args.rounds} "
        f"ranks={args.ranks} dirichlet_alpha={args.dirichlet_alpha} lr={args.lr} "
        f"lora_scaling={args.lora_scaling}"
        + (f" rank_lr_map={RANK_LR_MAP}" if args.lr is None else "")
    )
    print(f"Rank distribution (round-robin over {rank_values}): "
          f"{client_ranks[:10]}...  mean={np.mean(client_ranks):.1f}")
    client_dataloaders = [get_loader(i) for i in range(args.num_clients)]
    base_model = build_base_model(num_labels=num_labels)

    start = time.time()
    train_stacking(base_model, client_dataloaders)
    elapsed = time.time() - start
    logger.info(f"Total wall time: {elapsed / 3600:.2f} h")
    writer.close()
