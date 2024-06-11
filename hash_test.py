import torch

# 线性同余生成器 (LCG)
def lcg(seed, a=1664525, c=1013904223, m=2**32):
    return (a * seed + c) % m

# 第一个哈希函数版本
def Hash1_v1(key):
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

# 第二个哈希函数版本
def Hash1_v2(key):
    key = (~key) + (key << 21)
    key = key ^ (key >> 24)
    key = (key + (key << 3)) + (key << 8)
    key = key ^ (key >> 14)
    key = (key + (key << 2)) + (key << 4)
    key = key ^ (key >> 28)
    key = key + (key << 31)

    return key

# 生成随机数并计算碰撞
def generate_hashes(hash_func, n):
    keys = torch.randint(0, 2**32, (n,), dtype=torch.int64)
    hashes = [hash_func(key.item()) for key in keys]
    unique_hashes = set(hashes)
    return len(hashes) - len(unique_hashes)

# 生成 LCG 序列并计算碰撞
def generate_lcg_collisions(seed, n):
    sequence = []
    for _ in range(n):
        seed = lcg(seed)
        sequence.append(seed)
    unique_elements = set(sequence)
    return len(sequence) - len(unique_elements)

# 设置参数
n = 100000
seed = 1234

# 计算碰撞
collisions_lcg = generate_lcg_collisions(seed, n)
collisions_v1 = generate_hashes(Hash1_v1, n)
collisions_v2 = generate_hashes(Hash1_v2, n)

print(f"LCG 碰撞数: {collisions_lcg}")
print(f"Hash1_v1 碰撞数: {collisions_v1}")
print(f"Hash1_v2 碰撞数: {collisions_v2}")
