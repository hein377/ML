import numpy as np

def binary_to_decimal(binary):                  #binary = string, decimal = int
    binaryls, decimal = list(binary), 0
    for i in range(length:=len(binaryls)):
        val = int(binaryls[i])
        decimal += val * (2**(length-1-i))
    return decimal

def decimal_to_binary(decimal, num_bits):       #binary = string, decimal = int
    binary = ""
    for i in range(num_bits-1, -1, -1):
        if(decimal - 2**i >= 0): 
            binary += "1"
            decimal -= 2**i
        else: binary += "0"
    return binary

def create_truth_table(num_bits, canonical_int):            #returns [ ( np.array([[input]]).T , output ) <tuples of nx1 input_vector, output_value> ]; e.g. if num_bits=3: [ (np.array([[1, 1, 1]]).T, 0), (np.array([[1, 1, 0]]).T, 1) ... ] from largest to smallest
    ind, binary_int = 2**num_bits - 1, list(decimal_to_binary(canonical_int, 2**num_bits))
    return [(np.array([[int(n) for n in list(decimal_to_binary(i, num_bits))]]).T,int(binary_int[i])) for i in range(ind, -1, -1)]

def pretty_print_tt(table):
    for inputs, output in table:
        for i in inputs: print(i, end = "  ")
        print(f"|  {output}")

pretty_print_tt(create_truth_table(5, 35144000))