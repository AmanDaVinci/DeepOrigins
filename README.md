# DeepOrigins
Building the foundations of deep learning from matrix multiplication and backpropagation to ResNets and beyond
___

### Contents:
---
1. Matrix Multiplication  
2. Neural Network Forward Pass
3. Neural Network Backpropagation
4. Rebuilding PyTorch Essentials
___

### Notebooks:
---

## 01_Matrix-Multiplication.ipynb:  
Optimizing matrix multiplication from scratch with:
1. Nested loops in standard Python
2. Array Slicing
3. Array Broadcasting
4. Einstein Summation in PyTorch
5. Standard PyTorch
   
*Matrix multiplication in standard PyTorch is about 44,000 times faster than using standard python*
___

## 02_Neural-Network-Forward-Pass.ipynb:  
Why is it difficult to train feedforward neural networks?
1. Exploding Activations in just 4 layers (Solution: Xavier Initialization)
2. Vanishing Activations when using ReLU (Solution: Kaiming Initialization)
3. Improving with Shifted ReLU

Finally, we build a feed forward neural network mimicking PyTorch's modular API design with inbuilt Kaiming initialization for linear layers.
___

## 03_Neural-Network-Backpropagation.ipynb:  
Neural Network backpropagation in code from scratch
___

## 04_Rebuilding-PyTorch-Internals.ipynb:
Exploring and rebuilding the essential PyTorch internals for training neural networks
___
