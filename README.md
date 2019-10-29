## What is a neural network?
An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.

Each node has a bias and weight(s). Bias is an extra input to neurons and it is always 1, and has it’s own connection weight. This makes sure that even when all the inputs are none (all 0’s) there’s gonna be an activation in the neuron.


Y = sigma(weight * input) + bias

<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/an.png"/>



## Types of Activation Functions
The main purpose of an activation function is to convert an input signal of a node in a artificial neural network to an output signal. That output signal now is used as an input in the next layer in the stack.

1. Sigmoid or Logistic 
f(x) = 1 / 1 + exp(-x)
This function is usually used in the output layer, as adapts the output to range from 0 to 1(probability).
2. Tanh — Hyperbolic tangent
3. ReLu -Rectified linear units
Should be used only in hidden layers of neural network.

Reference
https://en.wikipedia.org/wiki/Artificial_neural_network
https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
https://hackernoon.com/everything-you-need-to-know-about-neural-networks-8988c3ee4491
https://www.youtube.com/watch?v=-wW9LMRXPsE
