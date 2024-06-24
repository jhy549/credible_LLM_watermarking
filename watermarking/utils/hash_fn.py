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


# def Hash1(key, shifts):
#     key = (~key) + (key << shifts[0])
#     key = key ^ (key >> shifts[1])
#     key = (key + (key << shifts[2])) + (key << shifts[3])
#     key = key ^ (key >> shifts[4])
#     key = (key + (key << shifts[5])) + (key << shifts[6])
#     key = key ^ (key >> shifts[7])
#     key = key + (key << shifts[8])

#     key = (key + (key << shifts[9])) + (key << shifts[10])
#     key = key ^ (key >> shifts[11])
#     key = (key + (key << shifts[12])) + (key << shifts[13])
#     key = key ^ (key >> shifts[14])
#     return key

# def random_shuffle(key, n, device) -> torch.Tensor:
#     idxs = torch.arange(n, device=device)
#     if not (isinstance(key, int)) and len(key) > 1:
#         key = key.unsqueeze(1)
#         idxs = idxs.unsqueeze(0)
    
#     # 固定的位移值模板
#     shifts = [21, 24, 3, 8, 14, 2, 4, 28, 31, 3, 8, 14, 2, 4, 28]
#     # 调整位移值
    
#     values = Hash1(key * n + idxs, shifts)
#     idxs = torch.sort(values, stable=True, dim=-1)[1]
#     return idxs