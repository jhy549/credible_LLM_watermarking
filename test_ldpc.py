import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message

# 定义原始消息
v = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])

# 参数设置
n = 60  # 码字长度，尝试选择一个能被d_c整除的值
d_v = 2  # 每个校验节点的连接数
d_c = 5  # 每个变量节点的连接数
snr = 100  # 非常高的信噪比，模拟无噪声情况

# 确保d_c能整除n
if n % d_c != 0:
    n = (n // d_c + 1) * d_c  # 调整n为d_c的倍数

# 生成LDPC矩阵
H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]  # 消息长度

# 确保消息长度与输入长度匹配
if len(v) > k:
    raise ValueError("消息长度大于生成矩阵允许的长度")
elif len(v) < k:
    # 如果v短于k，需要填充零
    v_padded = np.zeros(k, dtype=int)
    v_padded[:len(v)] = v
    v = v_padded

# 编码消息
y = encode(G, v, snr)
print(y)
# 受到攻击后的编码序列
y_attacked = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,0,0,0,0,1])

# 解码消息
d = decode(H, y_attacked, snr, maxiter=100)
v_decoded = get_message(G, d)

print("原始消息:", v)
print("受到攻击后的编码序列:", y_attacked)
print("解码消息:", v_decoded)
print("解码是否成功:", np.array_equal(v, v_decoded))
