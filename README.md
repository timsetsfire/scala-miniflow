# scala-miniflow

Came across this idea during my Udacity Deep Learning Nanodegree and thought it was very cool.
The original library, MiniFlow, was a neural network library written from scratch (in Python)
that behaves much like TensorFlow, Google's deep learning library.

I thought it would be fun to do something similar in Scala

See
(https://medium.com/udacity/the-miniflow-lesson-929200f72e27) and (https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd)
for more information.  

## Requirements

[SBT](www.scala-sbt.org)

## Usage

To package into jar

`sbt package`

For REPL

`sbt console`

## Supported Activation functions

* Sigmoid
* SoftMax
* Tanh
* ReLU
* LeakyReLU
* Maxout

## Supported Cost Functions

* Cross Entropy With Logits
* Binary Cross Entropy and Binary Cross Entropy With Logits
* Mean Square Error

For the Cross Entropy With Logits, I did this out of laziness.  

## Regularization

* Dropout

## Optimizers

* Adam
* Gradient Descent

## Misc

* Batch normalization is implemented and seems to work

## Main

The main method runs a simple GAN on the Mnist Test dataset and returns sample generated images.  Before you run the main method, get the mnist test dataset

`curl -s https://pjreddie.com/media/files/mnist_test.csv > resources/mnist_test.csv`

Run the main method in terminal via

`sbt "run n"` where `n` is the number of epochs.  

## Notebooks

To use this in a notebook

### Create jar

`sbt package`

will create ./target/scala-2.11/scala-miniflow_2.11-0.1.0-SNAPSHOT.jar

### Install Beakerx

install [beakerx](www.beakerx.comhttp://beakerx.com/documentation#tutorials-and-examples).  

```
conda create -y -n beakerx 'python>=3'
source activate beakerx
conda config --env --add pinned_packages 'openjdk>8.0.121'
conda install -y -c conda-forge ipywidgets beakerx
```

when you are in `beakerx` environment run

`jupyter notebook`

## Quick Start

See `notebooks/Quick Start.ipynb`
