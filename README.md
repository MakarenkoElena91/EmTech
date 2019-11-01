## How to run?
1. install anaconda
2. clone this repository
3. open terminal in the current folder and run the following command:
```jupyter notebook ```
or
```jupyter lab```
4. it will automatically run in the default browser or copy the url and paste it in the browser 
## What is a neural network?
An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.

A neuron is a weighted sum of all of its inputs(pixels) + bias and this sum is "fed" through an activation function. Bias is an extra input to neurons and it is always 1, and has it’s own connection weight. This makes sure that even when all the inputs are none (all 0’s) there’s gonna be an activation in the neuron.

<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/equation.png"/>
<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/an.png"/>


##Data
Input of 10 neurons- as we have 10 numbers from 0 till 9. All images are of 28x28 px size, which is 784 "flattened" pixels.
Matrix multiplication: First matrix (100x784): 100 images, one per line by 784 pixels. Second matrix(784x10) 784 pixels by 10 biases
                                                     
                                                     L = X.W + b
As we can't multiply these two matrixes, we need an activation function, which normalizes line by line

                                                      Y = softmax(X.W + b)
where:
y Predictions y[100, 10]

x Images x[100, 784]

w Weights w[784, 10]

b Biases b[10]
Initially weights and biases are assigned randomly.
## Types of Activation Functions
The main purpose of an activation function is to convert an input signal of a node in a artificial neural network to an output signal. That output signal now is used as an input in the next layer in the stack.

1. Sigmoid or Logistic 
f(x) = 1 / 1 + exp(-x)
<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/sigmoid.png"/>

This function is usually used in the output layer, as adapts the output to range from 0 to 1(probability).
2. Tanh — Hyperbolic tangent
3. ReLu -Rectified linear units
Should be used only in hidden layers of neural network.
4. Softmax 

Reference

https://en.wikipedia.org/wiki/Artificial_neural_network
https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
https://hackernoon.com/everything-you-need-to-know-about-neural-networks-8988c3ee4491
https://www.youtube.com/watch?v=-wW9LMRXPsE
