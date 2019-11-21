## How to run?
Run backend:
1. install anaconda
In order to install it on linux run the following commands in terminal:
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh
2. clone this repository
3. open terminal in the current folder and run the following command:
```jupyter notebook ```
or
```jupyter lab```
4. it will automatically run in the default browser or copy the url and paste it in the browser 

Run frontend:

 ```env FLASK_APP=webapp.py flask run```
 * Serving Flask app "webapp"
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

## What is a neural network?
An artificial neural network is an interconnected group of nodes, inspired by a simplification of neurons in a brain. Here, each circular node represents an artificial neuron and an arrow represents a connection from the output of one artificial neuron to the input of another.

A neuron is a weighted sum of all of its inputs(pixels) + bias and this sum is "fed" through an activation function. Bias is an extra input to neurons and it is always 1, and has it’s own connection weight. This makes sure that even when all the inputs are none (all 0’s) there’s gonna be an activation in the neuron.

<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/equation.png"/>
<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/an.png"/>
2 main types of neural network 
Conv2D work well on images and Conv1D on text.

## Data
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.
It is a dataset of 4 files that contains:
1. 60,000 images in the training dataset 
2. 10,000 images in the test dataset
3. 60,000 labels in the training dataset 
4. 10,000 labels in the test dataset

All images are small square 28×28 pixel grayscale, pre-aligned images of handwritten single digits between 0 and 9.

1. Load data
First step, we load the images and reshape the data arrays to have a single color channel.

We also know that there are 10 classes and that classes are represented as unique integers.

We can, therefore, use a one hot encoding for the class element of each sample, transforming the integer into a 10 element binary vector with a 1 for the index of the class value, and 0 values for all other classes. We can achieve this with the to_categorical() utility function.

We know that the pixel values for each image in the dataset are unsigned integers in the range between black and white, or 0 and 255.

2. Prepare Pixel Data
We need to normalize the pixel values of grayscale images, e.g. rescale them to the range [0,1]. This involves first converting the data type from unsigned integers to floats, then dividing the pixel values by the maximum value.

3. Define & Train Model
Input of 10 neurons- as we have 10 numbers from 0 till 9. 
All images are of 28x28 px size, which is 784 "flattened" pixels.
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

1. Softmax output to range from 0 to 1
2. Sigmoid or Logistic 
f(x) = 1 / 1 + exp(-x)
<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/sigmoid.png"/>

This function is usually used in the output layer, as adapts the output to range from 0 to 1(probability).
3. ReLu -Rectified linear units. 0 for all negative numbers and identity for all positive numbers.
Should be used only in hidden layers of neural network.

<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/relu.png"/>
4. Save and Test Model


Reference

https://en.wikipedia.org/wiki/Artificial_neural_network
https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
https://hackernoon.com/everything-you-need-to-know-about-neural-networks-8988c3ee4491
https://www.youtube.com/watch?v=-wW9LMRXPsE

https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8
