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
Optimizing matrix multiplication from scratch:
1. Nested loops in standard Python
2. Array Slicing
3. Array Broadcasting
4. Einstein Summation in PyTorch
5. Standard PyTorch
   
*Matrix multiplication in standard PyTorch is about 44,000 times faster than using standard python*
___

## 02_Neural-Network-Forward-Pass.ipynb:  
Demonstrating the difficulty in training neural networks:
1. Exploding Activations with added depth [Solution: Xavier Initialization]
2. Vanishing Activations when using ReLU [Solution: Kaiming Initialization]
3. Improvements with Parametric/Leaky/Shifted ReLU

After developing an appreciation of challenges in training neural networks, we build a Feed Forward Neural Network that mimics PyTorch's modular design.
___

## 03_Neural-Network-Backpropagation.ipynb:  
Implementing Autograd (Automatic Differentiation) functionality for:
1. Linear layer: Affine function
2. Activation layer: ReLU
3. Loss layer: Mean Squared Error

We design a layer abstraction class to build a Fully Connected Neural Network capable of backpropagating errors using automatic differentiation of its computation graph. PyTorch's design choices like nn.Module starts to make perfect sense.
___

## 04_Rebuilding-PyTorch-Internals.ipynb:
We explore the internal abstractions and architecture of PyTorch in depth and rebuild it from scratch:
1. PyTorch Data Abstractions
   1. Dataset
   2. DataLoader 
   3. DataSampler
2. PyTorch Training Abstractions
   1. nn.Parameters
   2. nn.Sequential
   3. Optimizer

After having dived deep into the inner workings of PyTorch, we gain a deeper understanding of deep learning concepts, the problems and the existing solutions. We get insight into the software architecture design and development process of a popular deep learning framework.
___
