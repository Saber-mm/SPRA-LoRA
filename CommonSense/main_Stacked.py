import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from copy import deepcopy
from arg import parse
import models
import random
from SPRALoRA import fl_stack_lora_train_het
from utils_data import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import math
# login("")
from accelerate import Accelerator


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model)


args = parse()

np.random.seed(args.dataset_seed)    
random.seed(args.dataset_seed)
torch.manual_seed(args.dataset_seed)

accelerator = Accelerator(cpu=False)

def generate_ranks(N, a, b, dist_type='uniform', seed=42):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if dist_type == 'uniform':
        # samples = np.random.uniform(a, b, N)

        exp_a = int(math.log2(a))
        exp_b = int(math.log2(b))
        exponent_range = list(range(exp_a, exp_b + 1))
        sampled_exps = random.choices(exponent_range, k=N)
        return [2**exp for exp in sampled_exps]

    elif dist_type == 'normal':
        mu = (a + b) / 2
        sigma = (b - a) / 6  # ~99.7% within [a, b]
        samples = np.random.normal(mu, sigma, N)

    elif dist_type == 'heavy_tail_low':
        # Use inverse log-normal to bias toward low values
        mu = np.log((a + b) / 4)
        sigma = 1.0
        samples = 1 / np.random.lognormal(mu, sigma, N)
        samples = a + (samples - np.min(samples)) / (np.max(samples) - np.min(samples)) * (b - a)
        
    else:
        raise ValueError("Unsupported distribution type.")

    samples = np.clip(samples, a, b)
    return np.round(samples).astype(int).tolist() 

print(f"Total number of GPUs (processes): {accelerator.num_processes}")
print(f"Current process index: {accelerator.process_index}")

r_list = generate_ranks(args.clients, args.lora_r_min, args.lora_r_max, dist_type=args.rank_type, seed=args.dataset_seed)
r_list = [2**math.floor(math.log2(n)) if n > 0 else 0 for n in r_list]
model = models.build_model_unquantized(args.base_model)
client_dataloaders = build_datasets(args, alpha=args.alpha_dirichlet)

fl_stack_lora_train_het(model, client_dataloaders,
    client_lr=args.client_lr,
    r_list=r_list,
    accelerator=accelerator,
    base_seed=args.dataset_seed
)