import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np
import shutil
import models
import time



args = parse()     

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


def sum_delta(client_model, server_lora_params, norm_sum, sum_delta_params, r_max, accelerator):
    device = accelerator.device
    if sum_delta_params == None:
        sum_delta_params = {}
        norm_sum = {}
    # client_lora_params = {normalize_name(n): cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    client_lora_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
    
    for name, param in client_lora_params.items():
        param = accelerator.reduce(param, reduction="mean")
        if "lora_B" in name or "lora_A" in name:
            client_rank = param.shape[1] if "lora_B" in name else param.shape[0]
            if "lora_B" in name:
                padded_delta_param = torch.zeros(param.data.shape[0], r_max, device=device)
                server_param = server_lora_params[name].to(param.device) # moving this param to gpu
                # client's delta param:
                padded_delta_param[:, :client_rank] = param.detach() - server_param[:, :client_rank].detach() 
            elif "lora_A" in name:
                padded_delta_param = torch.zeros(r_max, param.data.shape[1], device=device)
                server_param = server_lora_params[name].to(param.device) # moving this param to gpu
                # client's delta param:
                padded_delta_param[:client_rank, :] = param.detach() - server_param[:client_rank, :].detach()
        else:
            server_param = server_lora_params[name].to(param.device) # moving this param to gpu
            # client's delta param:
            padded_delta_param =  param.detach() - server_param.detach()
        
        if name not in sum_delta_params:
            sum_delta_params[name] = padded_delta_param.clone()
            norm_sum[name] = 1
        else: 
            sum_delta_params[name] = sum_delta_params[name] + padded_delta_param
            norm_sum[name] += 1 
    
    return norm_sum, sum_delta_params

def _set_lora_trainability(client_model, train_a, train_b):
    # Keep non-LoRA parameters untouched and only switch the LoRA matrices.
    for name, param in client_model.named_parameters():
        normalized_name = normalize_name(name)
        if "lora_A" in normalized_name:
            param.requires_grad = train_a
        elif "lora_B" in normalized_name:
            param.requires_grad = train_b


def fl_cmu_lora_train_het(frozen_model, client_dataloaders, server_optimizer, server_lr, client_lr, eval_freq, r_max, r_list, accelerator, base_seed):
    start = time.perf_counter()

    server_peft_model = models.add_adapter(frozen_model, r_max, args.lora_alpha)
    server_peft_model = server_peft_model.to(accelerator.device) # we use it with clients' models to calculate delta params, so we move it to gpu
    server_lora_params = {n:p for n, p in server_peft_model.named_parameters() if p.requires_grad == True}
    if server_optimizer == 'sgd':
        server_opt = torch.optim.SGD(server_lora_params.values(), lr=server_lr)
    elif server_optimizer == 'adam':
        server_opt = torch.optim.AdamW(server_lora_params.values(), lr=server_lr)
    else:
        raise ValueError()
    server_opt.zero_grad()

    if args.algorithm not in ["HetLoRA", "RoLoRA", "LA-LoRA"]:
        raise ValueError(f"Unsupported algorithm for fl_cmu_lora_train_het: {args.algorithm}")

    pbar = tqdm(range(args.num_comm_rounds))
    for rnd in pbar:
        print(f'communication round: {rnd}')
        sum_delta_params = None
        norm_sum = None
        client_ids = torch.randperm(args.clients)[:args.server_batch]            
        for i, client_id in enumerate(client_ids):
            # starting from the same initial pretrained model (no merging)
            client_model = models.add_adapter(deepcopy(frozen_model), r_list[client_id], args.lora_alpha)
            # initializing client's lora params with the server's updated lora params from the last round:   
            client_lora_params = {n: cp for n, cp in client_model.named_parameters() if cp.requires_grad==True}
            for n, sp in server_lora_params.items():
                if "lora_B" in n:
                    _, r_s = sp.shape
                    r_c = r_list[client_id] 
                    if r_c <= r_s:
                        client_lora_params[n].data = sp[:, :r_c].clone() # truncation
                    else:
                        raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                elif "lora_A" in n:
                    r_s, _ = sp.shape 
                    r_c = r_list[client_id] 
                    if r_c <= r_s:
                        client_lora_params[n].data = sp[:r_c, :].clone() # truncation
                    else:
                        raise ValueError(f"Client LoRA rank ({r_c}) exceeds server LoRA rank ({r_s}).")
                else:
                    # other trainable parameters: directly copy
                    client_lora_params[n].data = sp.clone()

            # HetLoRA keeps both matrices trainable; RoLoRA switches once per
            # round, while LA-LoRA switches inside the local-step loop.
            if args.algorithm == "RoLoRA":
                # Cross-round alternation: even rounds update B, odd rounds update A.
                _set_lora_trainability(client_model, train_a=(rnd % 2 == 1), train_b=(rnd % 2 == 0))

            client_model.config.use_cache = False  # Disable caching for training
            client_model = optimize_model_memory(client_model) # set train mode and set requires_grad=True
            #client_opt = torch.optim.SGD(client_model.parameters(), lr=client_lr, momentum=0.9)
            client_opt = torch.optim.AdamW(client_model.parameters(), lr=client_lr)
            client_opt.zero_grad()
            client_loader = client_dataloaders[client_id]
            client_model, client_opt, client_loader = accelerator.prepare(client_model, client_opt, client_loader)
            train_data_list = list(client_loader)
            for step in range(args.local_iter_per_round):
                if args.algorithm == "LA-LoRA":
                    # Alternate local updates within the round: even steps update B,
                    # odd steps update A.
                    _set_lora_trainability(client_model, train_a=(step % 2 == 1), train_b=(step % 2 == 0))
                batch = random.choice(train_data_list)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = client_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                client_opt.step()
                client_opt.zero_grad()

            if args.algorithm == "LA-LoRA":
                # Before aggregation, expose both LoRA matrices so that sum_delta(...)
                # collects the final A and B updates from this client. For RoLoRA only 
                # one of the matrices is trainable and aggregated, but sum_delta(...) 
                # will still work correctly
                _set_lora_trainability(client_model, train_a=True, train_b=True)
            norm_sum, sum_delta_params = sum_delta(client_model, server_lora_params, norm_sum, sum_delta_params, r_max, accelerator) 
            
            # Clean up after client training
            del client_model
            torch.cuda.empty_cache()
            accelerator.free_memory()
        
        # updating server params with the averaged delta params from clients:
        server_opt.zero_grad()
        for n, sp in server_lora_params.items():
            if args.algorithm == "HetLoRA":
                sp.grad = -1 * sum_delta_params[n] / norm_sum[n]
            elif n in sum_delta_params:
                # RoLoRA and LA-LoRA update only a subset of the LoRA tensors in each round.
                sp.grad = -1 * sum_delta_params[n] / norm_sum[n]
        server_opt.step()
        server_opt.zero_grad()
        
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
             # saving the final merged server model:
            if rnd == args.num_comm_rounds - 1:
                save_path = f"./model_parameters_set/{args.algorithm}_alpha_{args.alpha_dirichlet}_lr{client_lr}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd}"
                os.makedirs(save_path, exist_ok=True)
                if accelerator.is_main_process: 
                    raw_model = accelerator.unwrap_model(server_peft_model)
                    merged_model = raw_model.merge_and_unload()
                    merged_model.eval()
                    merged_model.save_pretrained(save_path)

        torch.cuda.empty_cache()

    end = time.perf_counter()
    duration = end - start
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    print(f"Duration: {hours}h {minutes}m")                  
