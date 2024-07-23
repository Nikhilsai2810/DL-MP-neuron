#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def mp_neuron(inputs, weights, threshold):
    """McCulloch-Pitts Neuron"""
    total_input = np.dot(inputs, weights)
    return 1 if total_input >= threshold else 0

def boolean_operations(operation):
    if operation == "OR":
        weights = [1, 1]
        threshold = 1
    elif operation == "AND":
        weights = [1, 1]
        threshold = 2
    elif operation == "NOR":
        weights = [-1, -1]
        threshold = -1
    elif operation == "NAND":
        weights = [-1, -1]
        threshold = -2
    elif operation == "NOT":
        weights = [-1]
        threshold = 0
    else:
        raise ValueError("Invalid operation")
    return weights, threshold

def generate_combinations(operation):
    if operation == "NOT":
        return [(0,), (1,)]
    else:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

def main():
    print("Available operations: OR, AND, NOR, NAND, NOT")
    operation = input("Enter the operation: ").strip().upper()

    # Get weights and threshold based on the operation
    weights, threshold = boolean_operations(operation)
    custom_weights = input(f"Enter custom weights (default {weights}): ")
    custom_threshold = input(f"Enter custom threshold (default {threshold}): ")

    if custom_weights:
        weights = list(map(int, custom_weights.split(',')))
    if custom_threshold:
        threshold = int(custom_threshold)

    # Generate input combinations
    combinations = generate_combinations(operation)

    print(f"Boolean Operation: {operation}")
    print(f"Weights: {weights}")
    print(f"Threshold: {threshold}")
    print("Input Combinations and Outputs:")
    
    for inputs in combinations:
        output = mp_neuron(inputs, weights, threshold)
        print(f"Inputs: {inputs} -> Output: {output}")

if __name__ == "__main__":
    main()


# In[3]:


import numpy as np

def mp_neuron(inputs, weights, threshold):
    """McCulloch-Pitts Neuron"""
    total_input = np.dot(inputs, weights)
    return 1 if total_input >= threshold else 0

def boolean_operations(operation):
    if operation == "OR":
        weights = [1, 1]
        threshold = 1
    elif operation == "AND":
        weights = [1, 1]
        threshold = 2
    elif operation == "NOR":
        weights = [-1, -1]
        threshold = -1
    elif operation == "NAND":
        weights = [-1, -1]
        threshold = -2
    elif operation == "NOT":
        weights = [-1]
        threshold = 0
    elif operation == "XOR":
        weights = [1, 1]
        threshold = 1
    else:
        raise ValueError("Invalid operation")
    return weights, threshold

def generate_combinations(operation):
    if operation == "NOT":
        return [(0,), (1,)]
    else:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

def main():
    print("Available operations: OR, AND, NOR, NAND, NOT, XOR")
    operation = input("Enter the operation: ").strip().upper()

    # Get weights and threshold based on the operation
    weights, threshold = boolean_operations(operation)
    custom_weights = input(f"Enter custom weights (default {weights}): ")
    custom_threshold = input(f"Enter custom threshold (default {threshold}): ")

    if custom_weights:
        weights = list(map(int, custom_weights.split(',')))
    if custom_threshold:
        threshold = int(custom_threshold)

    # Generate input combinations
    combinations = generate_combinations(operation)

    print(f"Boolean Operation: {operation}")
    print(f"Weights: {weights}")
    print(f"Threshold: {threshold}")
    print("Input Combinations and Outputs:")
    
    for inputs in combinations:
        output = mp_neuron(inputs, weights, threshold)
        print(f"Inputs: {inputs} -> Output: {output}")

if __name__ == "__main__":
    main()


# In[ ]:




