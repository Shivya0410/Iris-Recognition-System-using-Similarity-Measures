def lfsr(seed, taps, n_bits):
    """
    Generates a random string of bits using the LFSR method.
    
    Parameters:
    seed (int): The initial value of the LFSR, should be non-zero.
    taps (list): Positions where taps are applied for feedback.
    n_bits (int): Number of bits to generate.
    
    Returns:
    str: A string of bits of length n_bits.
    """
    lfsr = seed
    output = []
    for _ in range(n_bits):
        bit = sum((lfsr >> tap) & 1 for tap in taps) % 2  # Calculate the new bit from the taps
        output.append(str(bit))
        lfsr = (lfsr >> 1) | (bit << (len(bin(seed))-3))  # Shift right and insert new bit in MSB

    return ''.join(output)

# Example usage:
n = int(input("Enter the number of bits you want to generate: "))
# Example with seed and taps known to produce maximal length cycle for a 16-bit LFSR
random_bits = lfsr(seed=0xACE1, taps=[15, 14, 5, 4], n_bits=n)
print(f"Generated random bits: {random_bits}")
