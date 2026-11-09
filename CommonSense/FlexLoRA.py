import os
import torch
from tqdm import tqdm
from copy import deepcopy
from arg import parse
import random 
import numpy as np
import math
from copy import deepcopy
import models
import time

args = parse()

def normalize_name(name):
    while name.startswith("module."):
        name = name[len("module."):]  # Remove module. until no longer present
    return name

def weight_aggregation(client_peft_model, sum_scaled_deltaW, accelerator, r_c):
    if sum_scaled_deltaW == None:
        sum_scaled_deltaW = {}
    # client_lora_params = {normalize_name(n): cp for n, cp in client_peft_model.named_parameters() if cp.requires_grad == True}
    client_lora_params = {n: cp for n, cp in client_peft_model.named_parameters() if cp.requires_grad == True}
    for name, param in client_lora_params.items():
        client_lora_params[name] = accelerator.reduce(param, reduction="mean")
    
    for name, param in client_lora_params.items():
        if "lora_B" in name:
            paired_param_name = name.replace("lora_B", "lora_A")
            delta_W = torch.matmul(param.detach(), client_lora_params[paired_param_name].detach())
            s = 1 / math.sqrt(r_c) # scaling factor
            sum_scaled_deltaW[name] = sum_scaled_deltaW.get(name, torch.zeros_like(delta_W)) + s * delta_W.detach()
            
        if "lora_B" not in name and "lora_A" not in name:
            sum_scaled_deltaW[name] = sum_scaled_deltaW.get(name, torch.zeros_like(param)) + param.detach()             
            
    return sum_scaled_deltaW

def local_model_initialization(server_lora_params, sum_scaled_deltaW, num_clients, sum_ranks):
    U_set, S_set, V_set = {}, {}, {}
    for name, param in sum_scaled_deltaW.items():
        averaged_deltaW = (param / num_clients).detach() # uniform averaging
        
        if "lora_B" in name:
            U, S, V = torch.linalg.svd(averaged_deltaW, full_matrices=False) # actually averaged_deltaW = USV^T, here V represents V^T. 
            # U, S, V = torch.svd_lowrank(averaged_deltaW) # actually averaged_deltaW = USV^T, here V represents V^T.
            U, S, V = U.detach(), S.detach(), V.detach()
            U_set[name] = U
            S_set[name] = S
            paired_param_name = name.replace("lora_B", "lora_A")
            V_set[paired_param_name] = V

            _, r = server_lora_params[name].shape
            s_server = 1 / math.sqrt(r) # scaling factor

            # updating server model's LoRA parameters
            with torch.no_grad():
                server_lora_params[name].data.copy_( torch.matmul(U[:, :r], torch.diag(S[:r])) / s_server)
                server_lora_params[paired_param_name].data.copy_(V[:r,:])
            
        if "lora_B" not in name and "lora_A" not in name:
            with torch.no_grad():
                U_set[name] = averaged_deltaW
                # updating server model's parameter
                server_lora_params[name].data.copy_(averaged_deltaW)

    del averaged_deltaW        
    return U_set, S_set, V_set

def fl_svd_lora_train_het(frozen_model, client_dataloaders, client_lr, eval_freq, r_list, accelerator):
    start = time.perf_counter()

    server_peft_model = models.add_adapter(frozen_model, max(r_list), args.lora_alpha)
    server_lora_params = {n:p for n, p in server_peft_model.named_parameters() if p.requires_grad == True}

    torch.cuda.empty_cache()
    accelerator.free_memory()
    pbar = tqdm(range(args.num_comm_rounds))
    for rnd in pbar:
        print(f'Communication round: {rnd}')
        sum_scaled_deltaW = None
        client_ids = torch.randperm(len(client_dataloaders))[:args.server_batch]
        selected_sum = sum(r_list[i] for i in client_ids)
        for i, client_id in enumerate(client_ids):
            # starting from the same initial pretrained model (no merging)
            client_peft_model = models.add_adapter(deepcopy(frozen_model), r_list[client_id], args.lora_alpha)   
            client_lora_params = {n: cp for n, cp in client_peft_model.named_parameters() if cp.requires_grad==True}
            
            r_c = r_list[client_id]  # Client rank
            s_c = 1/math.sqrt(r_c)
            
            if rnd > 0: # for rnd>=1, initialize client lora params with SVD of updated "delta_W_g" before local training
                for n, cp in client_lora_params.items():
                    if "lora_B" in n:
                        B = torch.matmul(U_set[n][:, :r_c].detach(), torch.diag(S_set[n][:r_c].detach())) / s_c
                        if cp.data.shape == B.shape:
                            cp.data = B
                        else:
                            raise ValueError(f"svd mismatch B.")
                    elif "lora_A" in n:
                        A = V_set[n][:r_c,:].clone()
                        if cp.data.shape == A.shape:
                            cp.data = A
                        else:
                            print(cp.data.shape, A.shape)
                            raise ValueError(f"svd mismatch A.")
                    else:
                        # other trainable parameters: directly copy
                        cp.data = U_set[n].clone()


            client_peft_model.config.use_cache = False  # Disable caching for training
            client_opt = torch.optim.AdamW(client_peft_model.parameters(), lr=client_lr)
            #client_opt = torch.optim.SGD(client_peft_model.parameters(), lr=client_lr, momentum=0.9)
            client_opt.zero_grad()
            client_loader = client_dataloaders[client_id]
            client_peft_model, client_opt, client_loader = accelerator.prepare(client_peft_model, client_opt, client_loader)
            train_data_list = list(client_loader)
            for step in range(args.local_iter_per_round):
                batch = random.choice(train_data_list)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = client_peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)                    
                client_opt.step()
                client_opt.zero_grad()
            # summing scaled delta_W across clients:
            sum_scaled_deltaW = weight_aggregation(client_peft_model, sum_scaled_deltaW, accelerator, r_c)

            del client_peft_model
            del client_opt
            del client_loader
            torch.cuda.empty_cache()
            accelerator.free_memory()

        # to be passed to clients for the next round of local training. Server model is updated inside this function as well.
        U_set, S_set, V_set = local_model_initialization(server_lora_params, sum_scaled_deltaW, args.server_batch, selected_sum) # server_model will be updated inside
        
        torch.cuda.empty_cache()
        accelerator.free_memory()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # saving the final merged server model:
            if rnd == args.num_comm_rounds-1:
                save_path = f"./model_parameters_set/{args.algorithm}_lr{client_lr}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd}"
                os.makedirs(save_path, exist_ok=True)
                if accelerator.is_main_process: 
                    raw_model = accelerator.unwrap_model(server_peft_model)
                    merged_model = raw_model.merge_and_unload()
                    merged_model.eval()
                    merged_model.save_pretrained(save_path)

        torch.cuda.empty_cache()  
        accelerator.free_memory()          

    end = time.perf_counter()
    duration = end - start
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    print(f"Duration: {hours}h {minutes}m")                 

               
        
