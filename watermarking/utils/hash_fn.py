import torch
import random


def Hash1(key):
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)

    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    return key


def random_shuffle(key, n, device) -> torch.Tensor:
    idxs = torch.arange(n, device=device)
    if not (isinstance(key, int)) and len(key) > 1:
        key = key.unsqueeze(1)
        idxs = idxs.unsqueeze(0)
    values = Hash1(key * n + idxs)
    idxs = torch.sort(values, stable=True, dim=-1)[1]
    return idxs


def mersenne_twister(seed, n):
    random.seed(seed)
    return [random.randint(0, 2**32 - 1) for _ in range(n)]

def custom_shuffle(key, n, device) -> torch.Tensor:
    idxs = torch.arange(n, device=device)
    if not isinstance(key, int) and len(key) > 1:
        key = key.unsqueeze(1)
        idxs = idxs.unsqueeze(0)

    # 使用 LCG 生成确定性的伪随机数
    values = lcg(key * n + idxs)
    
    # 排序并返回打乱后的索引
    idxs = torch.sort(values, stable=True, dim=-1)[1]
    return idxs