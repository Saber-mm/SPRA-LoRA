import torch
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel  # noqa: F402
from transformers import BitsAndBytesConfig
#from utils_data import *


def build_model(base_model):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            token = True, 
            quantization_config=config,
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            cache_dir="HF_imported_pretrained_model"
        )
    model = prepare_model_for_kbit_training(model)
    return model

    

def build_model_unquantized(base_model):
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            token = True, 
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            cache_dir="HF_imported_pretrained_model"
        )
    return model



def add_adapter(model, r, alpha, target_modules= ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]): # ["q_proj", "v_proj"] comparison with FLoRA
    config = LoraConfig(
            r=r,
            lora_alpha=alpha, 
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights = "gaussian",
            use_rslora=True,
        )
    # print(seed)    
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)    
    model_adapter = PeftModel(model, config)
    return model_adapter






