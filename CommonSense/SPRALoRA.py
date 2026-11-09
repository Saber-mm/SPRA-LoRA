import os
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np
from peft.tuners.lora import LoraLayer
import shutil
import copy
import time
import models
import math

args = parse()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def optimize_model_memory(model):
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()
    return model

def normalize_name(name):
    while name.startswith("module."):
        name = name[len("module."):]  # Remove module. until no longer present
    return name

def fl_stack_lora_train_het(frozen_model, client_dataloaders, client_lr, r_list, accelerator, base_seed):
    start = time.perf_counter()

    lr_dict = {4:2e-4, 8:4e-4, 16:8e-4, 32:2e-3, 64:2e-3, 128:4e-3, 256:4-3} # from federated grid search

    target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    client_draws = [torch.randperm(len(r_list), generator=torch.Generator().manual_seed(args.dataset_seed+i))[:args.server_batch]
                    for i in range(args.num_comm_rounds)]

    pbar = tqdm(range(args.num_comm_rounds))
    for rnd in pbar:
        client_ids = client_draws[rnd]
        selected_sum = sum([r_list[i] for i in client_ids])
        # updated merged "frozen_model":
        server_model = models.add_adapter(frozen_model, selected_sum, args.lora_alpha, target_modules) 
        server_model.train()
        server_lora_params = {n:p for n, p in server_model.named_parameters() if p.requires_grad == True}
        if server_lora_params is None:
            raise ValueError("no trainable parameters in server model")

        stack_index = 0
        for i, client_id in enumerate(client_ids):
            if args.algorithm in ['FFA']:
                set_seed(args.dataset_seed) # sampling the same (submatrices of) A for all client at each round
            else: # i.e. ['FLoRA', 'SPRA-LoRA']
                set_seed(rnd) # the same (submatrices of) A for all clients at each round, but different across rounds    
            # starting from the updated frozen_model (merging):
            client_model = models.add_adapter(deepcopy(frozen_model), r_list[client_id], args.lora_alpha, target_modules)
            client_lora_params = {normalize_name(n): cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
            if args.algorithm in ['FFA', 'SPRA-LoRA']: #freezing A 
                for n, p in client_model.named_parameters():
                    if "lora_A" in n:
                        p.requires_grad = False                 
            client_model.config.use_cache = False  # Disable caching for training             
            client_model = optimize_model_memory(client_model) # set train mode and set requires_grad=True
            if args.algorithm=='SPRA-LoRA':
                client_opt = torch.optim.AdamW(client_model.parameters(), lr=lr_dict[r_list[client_id]])
                # client_opt = torch.optim.AdamW(client_model.parameters(), lr=lr_dict[r_list[client_id]])
            else:
                client_opt = torch.optim.AdamW(client_model.parameters(), lr=client_lr)    
            client_opt.zero_grad()
            client_loader = client_dataloaders[client_id]
            client_model, client_opt, client_loader = accelerator.prepare(client_model, client_opt, client_loader)
            
            train_data_list = list(client_loader)
            for step in range(args.local_iter_per_round):
                batch = random.choice(train_data_list)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                client_opt.step()
                client_opt.zero_grad()   

            for n in client_lora_params.keys():
                client_lora_params[n] = accelerator.reduce(client_lora_params[n], reduction="mean")
        
            # Aggregating local adapters by stacking:
            s_server = 1 / math.sqrt(selected_sum) # scaling factor
            s_c = 1 / math.sqrt(r_list[client_id]) # scaling factor for client
            for n in server_lora_params.keys():
                if "lora_B" in n:
                    server_lora_params[n].data[:, stack_index:(stack_index+r_list[client_id])] = client_lora_params[n].data.detach()
                elif "lora_A" in n:
                    server_lora_params[n].data[stack_index:(stack_index+r_list[client_id]), :] = client_lora_params[n].data.detach() * (s_c / s_server) * (1 / args.server_batch)
                else:
                    server_lora_params[n].data += client_lora_params[n].data.detach() * (1 / args.server_batch)
        
            stack_index += r_list[client_id]
            if stack_index==selected_sum and i < args.server_batch-1:
                raise ValueError(f"LoRA stacking mismatch: got {stack_index}, expected {selected_sum}")
            
            # Clean up after client training
            del client_model
            torch.cuda.empty_cache()
            accelerator.free_memory()

        accelerator.wait_for_everyone()
        last_save_path = os.path.abspath(f"./model_parameters_set/{args.algorithm}_alpha_{args.alpha_dirichlet}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd-1}")
        if os.path.exists(last_save_path) and os.path.isdir(last_save_path):
            shutil.rmtree(last_save_path)
            print(f"Deleted last saved model of round {rnd-1} to save the current model of round {rnd}")
        
        save_path = os.path.abspath(f"./model_parameters_set/{args.algorithm}_alpha_{args.alpha_dirichlet}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd}")

        os.makedirs(save_path, exist_ok=True)
        if accelerator.is_main_process: 
            raw_model = accelerator.unwrap_model(server_model)
            merged_model = raw_model.merge_and_unload()
            merged_model.eval()
            merged_model.save_pretrained(save_path) 
            del merged_model, raw_model
            del server_model    

        accelerator.wait_for_everyone()
        frozen_model = models.build_model_unquantized(save_path) # loading for the next round
        torch.cuda.empty_cache() 

    end = time.perf_counter()
    duration = end - start
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    print(f"Duration: {hours}h {minutes}m")      
