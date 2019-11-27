## How to run?
Run backend:
1. install anaconda

In order to install it on linux run the following commands in terminal:

```wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh```

```bash Anaconda3-5.3.1-Linux-x86_64.sh```

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
In this project a convolutional neural network was used as it more accurate in comparison to sequential one. 
As we are working with images and Conv2D work well on images while Conv1D on text, in our case Conv2D is used.

# Backend
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

We know that the pixel values for each image in the dataset are float32 dtype in the range between black and white, or 0 and 255.

2. Prepare Pixel Data

Putting data_format to channel_first, you say that for every layer your tensor will have this shape: 
(batch, channels, height, width), but for channel_last you gonna have (batch, height, width, channels).
Both TensorFlow and Theano expects a four dimensional tensor as input. But where TensorFlow expects the 'channels' dimension as the last dimension (index 3, where the first is index 0) of the tensor – i.e. tensor with shape (samples, rows, cols, channels) – Theano will expect 'channels' at the second dimension (index 1) – i.e. tensor with shape (samples, channels, rows, cols). The outputs of the convolutional layers will also follow this pattern.


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

model.save("model.h5")

# Frontend

The main thing is to format the data you are sending to backend. It should match the data of the training set. 
As our canvas size is 200x200 and the image needs to be send in 28x28 size we face a couple of issues: 
image can be drawn somewhere in the corner or not properly centered and sized.

In order to solve these problems, a number of steps were taken:

## 1. Crop image

Function np.nonzero(img) returns a tuple of arrays, one for each dimension of arr, containing the indices of the non-zero elements in that dimension.
As we draw our image white and white is rgb(255, 255, 255) this function suits us perfectly well.

(np.nonzero(img)).min() finds the very first indices of the elements that are non-zero. 
This will give us the left and upper bounds.
(np.nonzero(img)).max() finds the very last indices of the elements that are non-zero. 
In order to get the bottom and right bounds we need subtract those values from the totla number of pixels.

## 2. Resize the image 20x20

The actual digit size in mnist dataset is 20x20 the rest 8 pixels is the white frame around it. 
Depending on whether the height of the image is bigger than width or the other way round, the image was resized to 20x20.
Ref:https://stackoverflow.com/a/57990437

## 3. Find the center of mass
Now when we have the correct "digit size" we neeed to find out where to place it on 28x28 blank image. The digit image won't always have exact 4 pixels frame at each side.
Center of mass is "mean value across each dimension". In other words - take all x coordinates and average them - and you got x coordinate of your "center of mass", the same for y.

How to calculate center of mass?
<img src="https://github.com/MakarenkoElena91/EmTech/blob/master/img/matrix.png"/>
1. Calculate the total number of black pixels (in our example it is 4).
2. Calculate the distance till every black pixel at each row.
3. Sum them up (in our example it is 1+3+0+2=6).
4. Divide the sum you got in point 3 by the number you got in point 1. That is x coordinate of our "center of mass".(in our example it is 6/4=1.5)
5. Calculate the distance till every black pixel at each column.
6. Sum them up (in our example it is 2+0+3+1=6).
7. Divide the sum you got in point 6 by the number you got in point 1. That is y coordinate of our "center of mass".
(in our example it is6/4=1.5)

## 4. Recenter and resize image to 28x28
Ref:https://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python

Calculate the top & left offset. As our image is 28x28, the center of it is (14, 14). In order to calculate top & left offset we need substract the center (x,y) of the new blank image and center (x, y) of the digit image. The calculated result (x, y) are the coordinates where digit image should be pasted into our new image.
## 5. Make it grayscale
## 6. Transform image to an array & reformat the array accordingly
Now when we have an image which matches the size and format of the training image, we can transform this image into an array of pixels as our neural network cant understand images, it works only with numbers. The way pixels are represented in the array also should match the training array. So we are doing the same manipulation we did in the backend: reshape array to 
(1, 28, 28, 1), where the first 1 is our image, 28 and 28 is the size and 1 is the channel. We also need to cast it to float 32 and divide by 255 - same way we did in the backend.

Reference:

https://en.wikipedia.org/wiki/Artificial_neural_network
https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
https://hackernoon.com/everything-you-need-to-know-about-neural-networks-8988c3ee4491
https://www.youtube.com/watch?v=-wW9LMRXPsE
https://medium.com/coinmonks/handwritten-digit-prediction-using-convolutional-neural-networks-in-tensorflow-with-keras-and-live-5ebddf46dc8
https://www.codesofinterest.com/2017/09/keras-image-data-format.html
