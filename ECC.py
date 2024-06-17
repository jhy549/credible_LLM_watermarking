class ECC:
    def __init__(self):
        pass

    def encode(self, data_bits):
        """Encodes 4-bit data into 7-bit Hamming code."""
        if len(data_bits) != 4:
            raise ValueError("Data bits should be a list of 4 bits.")

        # Initialize the 7-bit code with data bits and placeholders for parity bits
        code = [0] * 7
        code[2], code[4], code[5], code[6] = data_bits

        # Calculate parity bits
        code[0] = code[2] ^ code[4] ^ code[6]  # p1
        code[1] = code[2] ^ code[5] ^ code[6]  # p2
        code[3] = code[4] ^ code[5] ^ code[6]  # p4

        return code

    def decode(self, encoded_bits):
        """Decodes 7-bit Hamming code into 4-bit data, correcting a single-bit error if present."""
        if len(encoded_bits) != 7:
            raise ValueError("Encoded bits should be a list of 7 bits.")

        # Calculate parity check bits
        p1 = encoded_bits[0] ^ encoded_bits[2] ^ encoded_bits[4] ^ encoded_bits[6]
        p2 = encoded_bits[1] ^ encoded_bits[2] ^ encoded_bits[5] ^ encoded_bits[6]
        p4 = encoded_bits[3] ^ encoded_bits[4] ^ encoded_bits[5] ^ encoded_bits[6]

        # Calculate the error position
        error_position = p1 + (p2 << 1) + (p4 << 2)

        if error_position != 0:
            print(f"Error detected at position: {error_position}")
            # Correct the error
            encoded_bits[error_position - 1] ^= 1

        # Extract the original data bits
        data_bits = [encoded_bits[2], encoded_bits[4], encoded_bits[5], encoded_bits[6]]

        return data_bits