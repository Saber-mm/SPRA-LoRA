import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    default="7,8",     type=str)
    parser.add_argument('--base_model',    default='meta-llama/Llama-3.2-3B-Instruct',  type=str)
    parser.add_argument('--dataset',    default='commensense',  type=str)
    parser.add_argument('--rank_type',    default='uniform',  type=str)
    parser.add_argument('--clients',    default=20,       type=int)
    parser.add_argument('--dataset_seed',    default=0,       type=int)
    parser.add_argument('--server_batch',    default=4,       type=int)
    parser.add_argument('--server_opt',       default='adam',  type=str)
    parser.add_argument('--server_lr',        default=3e-4,    type=float) #3e-4
    parser.add_argument('--client_lr',        default=1e-4,    type=float) #5e-4
    parser.add_argument('--alpha_dirichlet',        default=2.0,    type=float)
    parser.add_argument('--client_batch',     default=16,      type=int)
    parser.add_argument('--test_batch',     default=16,      type=int)
    parser.add_argument('--local_iter_per_round',     default=20,      type=int)
    parser.add_argument('--num_comm_rounds',     default=100,      type=int)
    parser.add_argument('--lora_r',     default=64,      type=int)
    parser.add_argument('--lora_r_min',     default=4,      type=int)
    parser.add_argument('--lora_r_max',     default=128,      type=int)
    parser.add_argument('--lora_alpha', default=1, type=int)
    parser.add_argument('--sketching_ratio',     default=1,      type=float)
    parser.add_argument('--eval_freq',  default=20,         type=int)
    parser.add_argument('--algorithm',       default='SPRA-LoRA',  type=str)
    parser.add_argument('--compression_ratio',     default=1,      type=float)

    return parser.parse_args()
