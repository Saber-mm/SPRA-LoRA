"""Non-IID client split of a GLUE training set with a Dirichlet distribution."""
from collections import Counter
from typing import List, Tuple

import numpy as np
from datasets import load_dataset

# --data_name -> GLUE config name
GLUE_TASKS = {
    'sst2': 'sst2', 'QQP': 'qqp', 'MNLI': 'mnli', 'STS-B': 'stsb', 'WNLI': 'wnli',
    'RTE': 'rte', 'MRPC': 'mrpc', 'qnli': 'qnli', 'cola': 'cola',
}


def get_tag(data_name):
    """Labels of the GLUE training set, and the indices sorted by label."""
    train_dataset = load_dataset("glue", GLUE_TASKS[data_name])["train"]
    id2targets = [train_dataset[i]['label'] for i in range(len(train_dataset))]
    sort_index = np.argsort(np.array(id2targets))
    return id2targets, sort_index


def data_from_dirichlet(
        data_name: str,
        alpha_value: float,
        nums_cls: int,
        nums_wk: int,
        nums_sample: int,
        seed: int = 42  # ← 新增参数
) -> Tuple[List[List[int]], List[float]]:
    """
    Deterministic non-IID split with Dirichlet distribution.

    Args:
        data_name   : 数据集名称（用于 get_tag）
        alpha_value : Dirichlet α
        nums_cls    : 类别数
        nums_wk     : 客户端数
        nums_sample : 每客户端样本数
        seed        : 随机种子（默认 42）

    Returns:
        data_idx : list[list[int]]  – 每个客户端的样本索引
        std_list : list[float]      – 每个客户端标签计数的标准差
    """
    # --- 0. 独立随机数生成器，保证不污染全局 ---
    rng = np.random.default_rng(seed)

    # --- 1. 按类别建立索引池 ---
    id2targets, sorted_indices = get_tag(data_name)         # 自定义工具
    class_indices = [[] for _ in range(nums_cls)]
    for idx in sorted_indices:                              # O(N)
        class_indices[id2targets[idx]].append(idx)
    for lst in class_indices:                               # 仅打乱一次
        rng.shuffle(lst)
    ptr = np.zeros(nums_cls, dtype=int)                     # 每类指针

    # --- 2. 采 Dirichlet 概率矩阵 ---
    dirichlet_mat = rng.dirichlet([alpha_value] * nums_cls, size=nums_wk)

    # --- 3. 按客户端批量采样 ---
    data_idx: List[List[int]] = []
    for j in range(nums_wk):
        cls_samples = rng.choice(nums_cls, size=nums_sample, p=dirichlet_mat[j])
        client_indices = []
        for cls in cls_samples:
            p = ptr[cls]
            if p >= len(class_indices[cls]):                # 类别用尽 → 重洗
                rng.shuffle(class_indices[cls])
                ptr[cls] = 0
                p = 0
            client_indices.append(class_indices[cls][p])
            ptr[cls] += 1
        data_idx.append(client_indices)

    # --- 4. 统计标签分布标准差 ---
    std_list = []
    for indices in data_idx:
        labels = [id2targets[i] for i in indices]
        counts = np.bincount(labels, minlength=nums_cls)
        std_list.append(counts.std(ddof=0))

    # 可选：打印前 5 个客户端的标签直方图
    for idx, samples in enumerate(data_idx[:min(5, nums_wk)]):
        print(f'Client {idx} label histogram →',
              Counter([id2targets[i] for i in samples]))
    print('mean label std:', np.mean(std_list))
    return data_idx, std_list
