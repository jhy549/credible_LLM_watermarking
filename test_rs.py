import reedsolo
from reedsolo import RSCodec

class BinaryCodec:
    def __init__(self, ecc_length):
        # 初始化 Reed-Solomon 编解码器，ecc_length 是纠错码的长度
        self.rs = RSCodec(ecc_length)

    def bin_list_to_bytearray(self, bin_list):
        byte_array = bytearray()
        # 确保二进制列表长度是8的倍数
        while len(bin_list) % 8 != 0:
            bin_list.append(0)  # 可以选择填充0
        for i in range(0, len(bin_list), 8):
            byte = 0
            for bit in bin_list[i:i+8]:
                byte = (byte << 1) | bit
            byte_array.append(byte)
        return byte_array

    def bytearray_to_bin_list(self, byte_array):
        bin_list = []
        for byte in byte_array:
            # 获取当前字节的二进制表示，并确保它是8位长
            bin_list.extend([int(bit) for bit in '{:08b}'.format(byte)])
        return bin_list

    def encode(self, bin_list):
        # 转换为字节序列
        byte_arr = self.bin_list_to_bytearray(bin_list)
        print("Encoded bytearray:", byte_arr)
        
        # 编码
        encoded = self.rs.encode(byte_arr)
        return encoded

    def decode(self, encoded):
        # 解码
        try:
            decoded, _, _ = self.rs.decode(encoded)
            # print("Decoded bytearray:", decoded)
            decoded_bin_list = self.bytearray_to_bin_list(decoded)
            return decoded_bin_list
        except reedsolo.ReedSolomonError as e:
            print("解码失败:", e)
            return None

# 示例使用
codec = BinaryCodec(8)  # 使用8字节纠错码
original_bin_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 指定的二进制数据

# 编码
encoded_data = codec.encode(original_bin_list)
print("Encoded data:", encoded_data)

# 解码
decoded_bin_list = codec.decode(encoded_data)
print("原始二进制序列:", original_bin_list)
print("解码后的二进制序列:", decoded_bin_list if decoded_bin_list is not None else "解码错误")













# import reedsolo
# from reedsolo import RSCodec

# def bin_list_to_bytearray(bin_list):
#     byte_array = bytearray()
#     # 确保二进制列表长度是8的倍数
#     while len(bin_list) % 8 != 0:
#         bin_list.append(0)  # 可以选择填充0
#     for i in range(0, len(bin_list), 8):
#         byte = 0
#         for bit in bin_list[i:i+8]:
#             byte = (byte << 1) | bit
#         byte_array.append(byte)
#     return byte_array

# def bytearray_to_bin_list(byte_array):
#     bin_list = []
#     for byte in byte_array:
#         # 获取当前字节的二进制表示，并确保它是8位长
#         bin_list.extend([int(bit) for bit in '{:08b}'.format(byte)])
#     return bin_list


# def encode_decode(bin_list, ecc_length):
#     # 初始化 Reed-Solomon 编解码器，ecc_length 是纠错码的长度
#     rs = RSCodec(ecc_length)
    
#     # 转换为字节序列
#     byte_arr = bin_list_to_bytearray(bin_list)
#     print(byte_arr)
    
#     # 编码
#     encoded = rs.encode(byte_arr)
    
#     # 人为引入一些错误（可选）
#     encoded[0] ^= 0x01  # 在第一个字节中引入错误
    
#     # 解码
#     try:
#         decoded,_,_ = rs.decode(encoded)
#         print(decoded)
#         decoded_bin_list = bytearray_to_bin_list(decoded)
#         return decoded_bin_list
#     except reedsolo.ReedSolomonError as e:
#         print("解码失败:", e)
#         return None

# # 示例使用
# original_bin_list = [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]  # 指定的二进制数据
# decoded_bin_list = encode_decode(original_bin_list, 8)  # 使用8字节纠错码

# print("原始二进制序列:", original_bin_list)
# print("解码后的二进制序列:", decoded_bin_list if decoded_bin_list is not None else "解码错误")

