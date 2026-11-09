import os, math, time, random, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, SubsetRandomSampler
from datasets import load_dataset
from transformers import (
    RobertaForSequenceClassification, RobertaTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from tensorboardX import SummaryWriter
import argparse

from dirichlet_data import data_from_dirichlet

# ── argument parsing ───────────────────────────────────────────────────────────

def str2bool(v):
    return str(v).lower() == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--alg',         default='HetLoRA', type=str,
                    choices=['HetLoRA', 'RoLoRA', 'LA-LoRA', 'FedSA', 'FlexLoRA', 'FSLoRA'])
parser.add_argument('--data_name',   default='sst2',    type=str)
parser.add_argument('--num_clients', default=20,        type=int)
parser.add_argument('--client_frac', default=0.2,       type=float)
parser.add_argument('--rounds',      default=150,       type=int,  help='comm rounds')
parser.add_argument('--local_steps', default=20,        type=int,  help='local steps per round')
parser.add_argument('--batch_size',  default=16,        type=int)
parser.add_argument('--lr',          default=5e-4,      type=float, help='client optimizer lr')
parser.add_argument('--optimizer',   default='adamw',   type=str, choices=['adamw', 'sgd'],
                    help='client optimizer (adamw or sgd)')
parser.add_argument('--momentum',    default=0.9,       type=float, help='momentum for sgd optimizer')
parser.add_argument('--ranks',       default='1,2,4,8', type=str,
                    help='comma-separated client rank values, distributed round-robin; '
                         'the server uses the largest')
parser.add_argument('--dirichlet_alpha', default=0.1,       type=float, help='Dirichlet alpha')
parser.add_argument('--lora_alpha',  default=16,        type=int)
parser.add_argument('--lora_dropout',default=0.1,       type=float)
parser.add_argument('--gpu',         default='0',       type=str)
parser.add_argument('--eval_every',  default=5,         type=int,  help='eval every N rounds')
parser.add_argument('--output_dir',  required=True,     type=str,  help='directory for logs and TensorBoard events')
parser.add_argument('--seed',        default=42,        type=int)
parser.add_argument('--lora_scaling', default=0, type=float,
                    help='if set, overrides lora_alpha: each client uses lora_alpha = lora_scaling * r_c')
parser.add_argument('--model',       default='roberta_base', type=str,
                    choices=['roberta_base'],
                    help='backbone model')
parser.add_argument('--freeze_classifier', type=str2bool, default=True,
                    help='freeze the classifier head full-rank weights (classifier.dense.base_layer '
                         'and classifier.out_proj). LoRA on classifier.dense remains trainable. '
                         'Pass "false" to disable.')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_client_optimizer(params, lr):
    """Build the client optimizer selected by --optimizer (adamw or sgd), with default weight decay."""
    if args.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=args.momentum)
    return torch.optim.AdamW(params, lr=lr)

# ── reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(args.seed)

def client_lora_alpha(r):
    """Return lora_alpha for a client of rank r. Uses lora_scaling if specified."""
    if args.lora_scaling > 0:
        return args.lora_scaling * r
    return args.lora_alpha

# ── rank assignment ────────────────────────────────────────────────────────────

def assign_ranks(N, ranks):
    """Distribute N clients uniformly (round-robin) across the given rank values."""
    return [ranks[i % len(ranks)] for i in range(N)]

rank_values  = [int(r) for r in args.ranks.split(',')]
client_ranks = assign_ranks(args.num_clients, rank_values)
max_rank     = max(rank_values)   # server LoRA rank
clients_per_round = max(1, int(args.num_clients * args.client_frac))

# ── model & tokenizer ──────────────────────────────────────────────────────────

if args.model == 'roberta_base':
    MODEL_NAME   = 'FacebookAI/roberta-base'
    tokenizer    = RobertaTokenizer.from_pretrained(MODEL_NAME)
    TARGET_MODULES = ["query", "value", "key", "dense"]
    _TASK_TYPE   = TaskType.SEQ_CLS


def build_base_model(num_labels=2):
    if args.model == 'roberta_base':
        return RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    raise ValueError(f"Unknown model: {args.model}")


def add_adapter(base_model, r, lora_alpha=None):
    """Wrap a deepcopy of base_model with a rank-r LoRA adapter."""
    if lora_alpha is None:
        lora_alpha = args.lora_alpha
    cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=_TASK_TYPE,
        target_modules=TARGET_MODULES,
        init_lora_weights="gaussian",
        use_rslora=True
    )
    peft_model = get_peft_model(deepcopy(base_model), cfg)
    if args.freeze_classifier:
        # Freeze classifier head's full-rank weights (modules_to_save copies),
        # but keep LoRA params on classifier.dense trainable.
        for n, p in peft_model.named_parameters():
            if "classifier" in n and "lora_" not in n and p.requires_grad:
                p.requires_grad = False
    return peft_model


# ── dataset loading ────────────────────────────────────────────────────────────

GLUE_CONFIGS = {
    'sst2':  dict(task='sst2',  text1='sentence',   text2=None,        num_labels=2,  val_split='validation'),
    'cola':  dict(task='cola',  text1='sentence',   text2=None,        num_labels=2,  val_split='validation'),
    'mrpc':  dict(task='mrpc',  text1='sentence1',  text2='sentence2', num_labels=2,  val_split='validation'),
    'rte':   dict(task='rte',   text1='sentence1',  text2='sentence2', num_labels=2,  val_split='validation'),
    'qnli':  dict(task='qnli',  text1='question',   text2='sentence',  num_labels=2,  val_split='validation'),
    'qqp':   dict(task='qqp',   text1='question1',  text2='question2', num_labels=2,  val_split='validation'),
    'mnli':  dict(task='mnli',  text1='premise',    text2='hypothesis', num_labels=3, val_split='validation_matched'),
}

_glue_cfg = GLUE_CONFIGS[args.data_name.lower()]

def _preprocess(examples):
    if _glue_cfg['text2']:
        return tokenizer(examples[_glue_cfg['text1']], examples[_glue_cfg['text2']],
                         truncation=True, padding='max_length', max_length=128)
    return tokenizer(examples[_glue_cfg['text1']],
                     truncation=True, padding='max_length', max_length=128)

_raw      = load_dataset("glue", _glue_cfg['task'])
tokenized = _raw.map(_preprocess, batched=True)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataset = tokenized['train']
test_dataset  = tokenized[_glue_cfg['val_split']]
num_labels    = _glue_cfg['num_labels']


# ── data partitioning ──────────────────────────────────────────────────────────

_nums_sample = len(train_dataset) // args.num_clients
data_idx, _  = data_from_dirichlet(args.data_name, args.dirichlet_alpha,
                                    num_labels, args.num_clients, _nums_sample)


def get_loader(pid):
    sampler = SubsetRandomSampler(data_idx[pid])
    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                      num_workers=0)

def get_test_loader():
    return DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ── evaluation ─────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            tgt    = batch['label'].to(device)
            logits = model(ids, attention_mask=mask).logits
            correct += (logits.argmax(dim=-1) == tgt).sum().item()
            total   += tgt.size(0)
    model.to('cpu')
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

# ── helper: copy server LoRA params to client at truncated rank ────────────────

def init_client_from_server(client_params, server_params, r_c):
    """Copy server LoRA slices of rank r_c into client model parameters."""
    for n, cp in client_params.items():
        if n not in server_params:
            continue
        sp = server_params[n]
        if 'lora_B' in n:
            cp.data.copy_(sp[:, :r_c].detach())
        elif 'lora_A' in n:
            cp.data.copy_(sp[:r_c, :].detach())
        else:
            cp.data.copy_(sp.detach())

# ── helper: local training ─────────────────────────────────────────────────────

def set_lora_trainability(model, train_a, train_b):
    """Switch requires_grad on the LoRA A / B matrices; other params are untouched."""
    for n, p in model.named_parameters():
        if 'lora_A' in n:
            p.requires_grad = train_a
        elif 'lora_B' in n:
            p.requires_grad = train_b

def local_train(model, loader, grad_hook=None, step_hook=None):
    """Run --local_steps optimizer steps on random batches from loader.
    grad_hook(model), if given, runs after backward() and before the step.
    step_hook(model, step), if given, runs before each step's forward pass."""
    model.train()
    model.to(device)
    opt = make_client_optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr)
    data_list = list(loader)
    for step in range(args.local_steps):
        if step_hook is not None:
            step_hook(model, step)
        batch = random.choice(data_list)
        opt.zero_grad()
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        loss = criterion(model(ids, attention_mask=mask).logits, batch['label'].to(device).long())
        loss.backward()
        if grad_hook is not None:
            grad_hook(model)
        opt.step()
    model.to('cpu')

def sample_clients():
    return np.random.choice(args.num_clients, size=clients_per_round, replace=False)


def log_eval(server_model, test_loader, rnd, best_acc, acc=None):
    """Evaluate on rounds divisible by --eval_every; return the updated best accuracy.
    If acc is given, log it instead of evaluating server_model."""
    if rnd % args.eval_every != 0:
        return best_acc
    if acc is None:
        acc = evaluate(server_model, test_loader)
    best_acc = max(best_acc, acc)
    writer.add_scalar('accuracy', acc, rnd)
    logger.info(f"Round {rnd}: acc={acc:.2f}  best={best_acc:.2f}  alg={args.alg}  "
                f"data={args.data_name}  r_max={max_rank}  ranks={args.ranks}")
    print(f"[{args.alg}] Round {rnd}: acc={acc:.2f}")
    return best_acc

# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

# ── HetLoRA / RoLoRA / LA-LoRA (zero-padding) ───────────────────────────────

def pad_delta(n, cp, sp, r_c):
    """Client delta (cp - server slice), zero-padded back to the server shape."""
    if 'lora_B' in n:
        padded = torch.zeros_like(sp)
        padded[:, :r_c] = cp.detach() - sp[:, :r_c].detach()
    elif 'lora_A' in n:
        padded = torch.zeros_like(sp)
        padded[:r_c, :] = cp.detach() - sp[:r_c, :].detach()
    else:
        padded = cp.detach() - sp.detach()
    return padded

def la_lora_step_hook(model, step):
    """LA-LoRA: alternate inside the round — even steps update B, odd steps update A."""
    set_lora_trainability(model, train_a=(step % 2 == 1), train_b=(step % 2 == 0))

def run_hetlora(base_model):
    """
    HetLoRA: server keeps rank max_rank. Clients get truncated rank r_c.
    Client deltas are zero-padded back to max_rank and averaged; rows a client
    did not train contribute zero delta, so they are left unchanged.
    RoLoRA:  same, but each round trains only B (even rounds) or only A (odd rounds).
    LA-LoRA: same, but alternates B / A at every local step within the round.
    """
    server_model = add_adapter(base_model, max_rank, lora_alpha=client_lora_alpha(max_rank))
    server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}

    test_loader = get_test_loader()
    best_acc = 0.0

    for rnd in range(args.rounds):
        client_ids = sample_clients()
        aggregate = {n: torch.zeros_like(p) for n, p in server_params.items()}

        for cid in client_ids:
            r_c = client_ranks[cid]
            client_base  = deepcopy(base_model)
            client_model = add_adapter(client_base, r_c, lora_alpha=client_lora_alpha(r_c))
            client_params = {n: p for n, p in client_model.named_parameters() if p.requires_grad}
            init_client_from_server(client_params, server_params, r_c)

            step_hook = None
            if args.alg == 'RoLoRA':
                set_lora_trainability(client_model, train_a=(rnd % 2 == 1), train_b=(rnd % 2 == 0))
            elif args.alg == 'LA-LoRA':
                step_hook = la_lora_step_hook   # optimizer holds both A and B; frozen ones get no grad

            local_train(client_model, get_loader(cid), step_hook=step_hook)

            # Frozen matrices are unchanged, so their delta is exactly zero.
            for n, cp in client_params.items():
                aggregate[n] += pad_delta(n, cp, server_params[n], r_c)

            del client_model, client_base
            torch.cuda.empty_cache()

        # Server update: apply averaged client delta to server params
        n_clients = max(len(client_ids), 1)
        with torch.no_grad():
            for n, sp in server_params.items():
                sp.data.add_(aggregate[n] / n_clients)

        best_acc = log_eval(server_model, test_loader, rnd, best_acc)

    logger.info(f"Final best accuracy: {best_acc:.2f}")


# ── FedSA (shared A, personal B) ──────────────────────────────────────────────

def run_fedsa(base_model):
    """
    FedSA: runs HetLoRA "in shadow" on the server, but clients only download A.
    Each client keeps its own B across rounds (first visit: B from the server).
    Reported accuracy = mean over this round's clients of (server A + own B)
    on the test set. The shadow server model's accuracy is logged separately.
    """
    server_model = add_adapter(base_model, max_rank, lora_alpha=client_lora_alpha(max_rank))
    server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}
    client_b = {}   # cid -> {param name: local B tensor (cpu)}

    def build_client(cid):
        r_c = client_ranks[cid]
        model = add_adapter(base_model, r_c, lora_alpha=client_lora_alpha(r_c))
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        init_client_from_server(params, server_params, r_c)
        if cid in client_b:
            for n, b in client_b[cid].items():
                params[n].data.copy_(b)
        return model, params

    test_loader = get_test_loader()
    best_acc = 0.0

    for rnd in range(args.rounds):
        client_ids = [int(c) for c in sample_clients()]
        aggregate = {n: torch.zeros_like(p) for n, p in server_params.items()}

        for cid in client_ids:
            r_c = client_ranks[cid]
            client_model, client_params = build_client(cid)

            local_train(client_model, get_loader(cid))

            client_b[cid] = {n: p.detach().cpu().clone()
                             for n, p in client_params.items() if 'lora_B' in n}
            for n, cp in client_params.items():
                aggregate[n] += pad_delta(n, cp, server_params[n], r_c)

            del client_model
            torch.cuda.empty_cache()

        n_clients = max(len(client_ids), 1)
        with torch.no_grad():
            for n, sp in server_params.items():
                sp.data.add_(aggregate[n] / n_clients)

        if rnd % args.eval_every == 0:
            accs = []
            for cid in client_ids:
                client_model, _ = build_client(cid)
                accs.append(evaluate(client_model, test_loader))
                del client_model
                torch.cuda.empty_cache()
            acc = float(np.mean(accs))
            server_acc = evaluate(server_model, test_loader)
            writer.add_scalar('accuracy_server_shadow', server_acc, rnd)
            best_acc = log_eval(server_model, test_loader, rnd, best_acc, acc=acc)
            logger.info(f"Round {rnd}: shadow server acc={server_acc:.2f}  "
                        f"per-client accs={[round(a, 2) for a in accs]}")

    logger.info(f"Final best accuracy: {best_acc:.2f}")


# ── FlexLoRA (SVD aggregation) ─────────────────────────────────────────────────

def run_flexlora(base_model):
    """
    FlexLoRA: aggregate sum(B_i @ A_i), run SVD, redistribute U·S, V to clients.
    """
    server_model = add_adapter(base_model, max_rank, lora_alpha=client_lora_alpha(max_rank))
    server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}
    r = max_rank

    # SVD state (initialised from server after first round)
    U_set, S_set, V_set = {}, {}, {}

    test_loader = get_test_loader()
    best_acc = 0.0

    for rnd in range(args.rounds):
        client_ids = sample_clients()
        aggregate  = {}   # name -> sum of (1/sqrt(r_c)) * B_c @ A_c

        for cid in client_ids:
            r_c = client_ranks[cid]
            client_base  = deepcopy(base_model)
            client_model = add_adapter(client_base, r_c, lora_alpha=client_lora_alpha(r_c))
            client_params = {n: p for n, p in client_model.named_parameters() if p.requires_grad}

            # Initialise from SVD state (skip round 0)
            if rnd > 0:
                for n, cp in client_params.items():
                    if 'lora_B' in n:
                        B = torch.mm(U_set[n][:, :r_c],
                                     torch.diag(S_set[n][:r_c]) * math.sqrt(r_c))
                        cp.data.copy_(B)
                    elif 'lora_A' in n:
                        cp.data.copy_(V_set[n][:r_c, :])
                    else:
                        cp.data.copy_(U_set[n])

            local_train(client_model, get_loader(cid))

            # Accumulate (1/sqrt(r_c)) * B_c @ A_c
            for n, cp in client_params.items():
                if 'lora_B' in n:
                    a_name = n.replace('lora_B', 'lora_A')
                    A_c = client_params[a_name].detach()
                    dW  = (1.0 / math.sqrt(r_c)) * torch.mm(cp.detach(), A_c)
                    aggregate[n] = aggregate.get(n, torch.zeros_like(dW)) + dW
                elif 'lora_A' in n:
                    pass  # handled via lora_B
                else:
                    aggregate[n] = aggregate.get(n, torch.zeros_like(cp)) + cp.detach()

            del client_model, client_base
            torch.cuda.empty_cache()

        # SVD decomposition and server update
        n_clients = max(len(client_ids), 1)
        for n, mat in aggregate.items():
            if 'lora_B' in n:
                U, S, Vh = torch.linalg.svd(mat / n_clients)
                U_set[n] = U.detach()
                S_set[n] = S.detach()
                a_name   = n.replace('lora_B', 'lora_A')
                V_set[a_name] = Vh.detach()
                with torch.no_grad():
                    server_params[n].data.copy_(
                        torch.mm(U[:, :r], torch.diag(S[:r]) * math.sqrt(r)))
                    server_params[a_name].data.copy_(Vh[:r, :])
            elif 'lora_A' in n:
                pass  # already handled
            else:
                U_set[n] = (mat / n_clients).detach()
                server_params[n].data.copy_(mat / n_clients)

        best_acc = log_eval(server_model, test_loader, rnd, best_acc)

    logger.info(f"Final best accuracy: {best_acc:.2f}")


# ── FSLoRA (Federated Sketching LoRA) ─────────────────────────────────────────

def run_fslora(base_model):
    """
    FSLoRA: each client applies a sketching mask selecting m out of r dimensions.
    Aggregate masked deltas; server updated via simple average.
    """
    server_model  = add_adapter(base_model, max_rank, lora_alpha=client_lora_alpha(max_rank))
    server_params = {n: p for n, p in server_model.named_parameters() if p.requires_grad}
    r = max_rank

    test_loader = get_test_loader()
    best_acc = 0.0

    for rnd in range(args.rounds):
        client_ids = sample_clients()
        aggregate  = None

        for cid in client_ids:
            m = client_ranks[cid]
            client_model  = deepcopy(server_model)
            client_params = {n: p for n, p in client_model.named_parameters() if p.requires_grad}

            # Build sketching matrices: select m random rows/cols from r dims
            sketching = {}
            mask_set  = {}
            rand_perm = torch.randperm(r)[:m]

            for n, p in client_params.items():
                S    = torch.ones_like(p.data)
                mask = torch.ones_like(p.data)
                if 'lora_B' in n:
                    S    = torch.zeros_like(p.data)
                    mask = torch.zeros_like(p.data)
                    mask[:, rand_perm] = 1
                    S[:, rand_perm]    = r / m
                    p.data *= S          # p = B * S  (algebraic sketch)
                elif 'lora_A' in n:
                    S    = torch.zeros_like(p.data)
                    mask = torch.zeros_like(p.data)
                    mask[rand_perm, :] = 1
                    S[rand_perm, :]    = r / m
                sketching[n] = S
                mask_set[n]  = mask

            # Local training with corrected gradient for lora_B
            def correct_lora_B_grad(model):
                for n, p in model.named_parameters():
                    if 'lora_B' in n and p.grad is not None:
                        p.grad *= sketching[n].to(device) ** 2

            local_train(client_model, get_loader(cid), grad_hook=correct_lora_B_grad)

            # Recover B from B*S
            for n, p in client_params.items():
                if 'lora_B' in n:
                    eps = 1e-8
                    p.data /= (sketching[n] + eps)

            # Compute masked delta and aggregate
            delta = {}
            for n, cp in client_params.items():
                sp  = server_params[n]
                delta[n] = (cp.detach() - sp.detach()) * mask_set[n]

            if aggregate is None:
                aggregate = delta
            else:
                for n in aggregate:
                    aggregate[n] = aggregate[n] + delta[n]

            del client_model
            torch.cuda.empty_cache()

        # Server update: apply averaged client delta to server params
        n_clients = max(len(client_ids), 1)
        with torch.no_grad():
            for n, sp in server_params.items():
                sp.data.add_(aggregate[n] / n_clients)

        best_acc = log_eval(server_model, test_loader, rnd, best_acc)

    logger.info(f"Final best accuracy: {best_acc:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info(
        f"alg={args.alg} data={args.data_name} num_clients={args.num_clients} "
        f"client_frac={args.client_frac} local_steps={args.local_steps} rounds={args.rounds} "
        f"r={max_rank} ranks={args.ranks} dirichlet_alpha={args.dirichlet_alpha} lr={args.lr}"
    )
    print(f"Rank distribution (round-robin over {rank_values}): {client_ranks[:10]}...  "
          f"mean={np.mean(client_ranks):.1f}")

    base_model = build_base_model(num_labels=num_labels)

    start = time.time()
    if args.alg in ('HetLoRA', 'RoLoRA', 'LA-LoRA'):
        run_hetlora(base_model)
    elif args.alg == 'FedSA':
        run_fedsa(base_model)
    elif args.alg == 'FlexLoRA':
        run_flexlora(base_model)
    elif args.alg == 'FSLoRA':
        run_fslora(base_model)

    elapsed = time.time() - start
    logger.info(f"Total wall time: {elapsed/3600:.2f} h")
    writer.close()
