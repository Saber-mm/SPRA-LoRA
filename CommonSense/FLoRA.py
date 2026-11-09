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

def fl_stack_lora_train_het(frozen_model, client_dataloaders, client_lr, m_list, accelerator, base_seed):
    start = time.perf_counter()

    pbar = tqdm(range(args.num_comm_rounds))
    for rnd in pbar:
        print('Communication round: ', rnd)
        client_ids = torch.randperm(len(client_dataloaders))[:args.server_batch]
        selected_sum = sum(m_list[i] for i in client_ids)
        # updated merged frozen_model:
        server_model = models.add_adapter(deepcopy(frozen_model), selected_sum, args.lora_alpha)   
        server_model.train()
        server_params = {n:p for n, p in server_model.named_parameters() if p.requires_grad == True}
        if server_params is None:
            raise ValueError("no trainable parameters in server model")
        
        stack_index = 0
        for i, client_id in enumerate(client_ids):
            set_seed(rnd) # the same (submatrices of) A for all clients at each round, but different across rounds 
            # starting from the updated frozen_model (merging):
            client_model = models.add_adapter(deepcopy(frozen_model), m_list[client_id], args.lora_alpha)   
            client_model.config.use_cache = False  # Disable caching for training             
            client_model = optimize_model_memory(client_model) # set train mode and set requires_grad=True
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
                
            ## Aggregating local adapters by stacking
            client_params = {normalize_name(n): cp for n, cp in client_model.named_parameters() if cp.requires_grad == True}
            for n in client_params:
                client_params[n] = accelerator.reduce(client_params[n], reduction="mean")
                
            for n in server_params:
                if "lora_B" in n:
                    server_params[n].data[:, stack_index:(stack_index+m_list[client_id])] = client_params[n].data.detach()
                elif "lora_A" in n:
                    server_params[n].data[stack_index:(stack_index+m_list[client_id]), :] = client_params[n].data.detach()/args.server_batch
                else: # classification head: just average them 
                    server_params[n].data = server_params[n].data + client_params[n].data.detach()/args.server_batch

            stack_index += m_list[client_id]
            if stack_index==selected_sum and i < args.server_batch-1:
                raise ValueError(f"LoRA stacking mismatch: got {stack_index}, expected {selected_sum}")
            
            # Clean up after client training
            del client_model
            torch.cuda.empty_cache()
            accelerator.free_memory()
        
        accelerator.wait_for_everyone()
        
        last_save_path = f"./model_parameters_set/{args.algorithm}_lr{client_lr}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd-1}"
        if os.path.exists(last_save_path) and os.path.isdir(last_save_path):
            shutil.rmtree(last_save_path)
            print(f"Successfully deleted last saved model of round {rnd-1} to save the current model of round {rnd}")
        
        save_path = f"./model_parameters_set/{args.algorithm}_lr{client_lr}_seed{args.dataset_seed}_T{args.local_iter_per_round}_rounds{args.num_comm_rounds}_type{args.rank_type}/{rnd}"
        os.makedirs(save_path, exist_ok=True)
        if accelerator.is_main_process: 
            raw_model = accelerator.unwrap_model(server_model)
            merged_model = raw_model.merge_and_unload()
            merged_model.eval()
            merged_model.save_pretrained(save_path)
            del merged_model, raw_model

        accelerator.wait_for_everyone()
        frozen_model = models.build_model_unquantized(save_path) # loading for the next round
        del server_model
        torch.cuda.empty_cache()
    
    end = time.perf_counter()
    duration = end - start
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    print(f"Duration: {hours}h {minutes}m")