def decimal_to_binary(n, bits):
    return bin(n)[2:].zfill(bits)

def hamming_encode(data):
    d = [int(bit) for bit in data]
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    return f"{p1}{p2}{d[0]}{p3}{d[1]}{d[2]}{d[3]}"

def encode_decimal(decimal):
    binary_str = decimal_to_binary(decimal, 7)
    encoded_str = ""
    for i in range(0, len(binary_str), 4):
        nibble = binary_str[i:i+4]
        if len(nibble) < 4:
            nibble = nibble.ljust(4, '0')
        encoded_str += hamming_encode(nibble)
    return encoded_str

def hamming_decode(encoded):
    bits = [int(bit) for bit in encoded]
    p1 = bits[0] ^ bits[2] ^ bits[4] ^ bits[6]
    p2 = bits[1] ^ bits[2] ^ bits[5] ^ bits[6]
    p3 = bits[3] ^ bits[4] ^ bits[5] ^ bits[6]
    error_position = p1 * 1 + p2 * 2 + p3 * 4

    if error_position != 0:
        bits[error_position - 1] ^= 1

    data_bits = f"{bits[2]}{bits[4]}{bits[5]}{bits[6]}"
    return data_bits

def binary_to_decimal(binary_str):
    return int(binary_str, 2)

def decode_encoded(encoded_str):
    decoded_str = ""
    for i in range(0, len(encoded_str), 7):
        encoded_nibble = encoded_str[i:i+7]
        decoded_str += hamming_decode(encoded_nibble)
    return binary_to_decimal(decoded_str)

def test_correcting_code(decimal_value, error_position):
    print(f"Original Decimal: {decimal_value}")
    encoded_str = encode_decimal(decimal_value)
    print(f"Encoded: {encoded_str}")

    # Introduce an error at the specified position
    received_str = list(encoded_str)
    received_str[error_position] = '1' if encoded_str[error_position] == '0' else '0'
    received_str = ''.join(received_str)
    print(f"Received (with error at position {error_position}): {received_str}")

    decoded_decimal = decode_encoded(received_str)
    print(f"Decoded: {decoded_decimal}")
    print(f"Error corrected: {decoded_decimal == decimal_value}")
    print("")

# 测试不同的十进制数和错误位置
test_correcting_code(25, 2)
test_correcting_code(25, 5)
test_correcting_code(25, 10)
test_correcting_code(25, 15)
test_correcting_code(100, 3)
test_correcting_code(100, 8)
test_correcting_code(100, 12)
test_correcting_code(100, 18)
