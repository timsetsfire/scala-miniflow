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

Create a new notebook using the Scala kernel.  Enter the following into cells and execute.  The first `%%classpath` command will download the ND4J requirements.  The second will add the neural network library.  

```
%%classpath add mvn
org.nd4j nd4j-native-platform 0.7.2
org.nd4j nd4s_2.11 0.7.2
```

```
%classpath add jar target/scala-2.11/scala-miniflow_2.11-0.1.0-SNAPSHOT.jar
```

## Quick Start

Suppose that `x_` and `y_` are INDArray of are the features and labels of the Boston Dataset.  

```:scala
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log,pow,sqrt}
import org.nd4s.Implicits._

import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.regularization._
import com.github.timsetsfire.nn.optimize._
import com.github.timsetsfire.nn.graph._

val x = new Input()
val y = new Input()
val h1 = ReLU(x, (13,6))
val yhat = Linear(h1, (6,1))
val mse = new MSE(y, yhat)
val network = topologicalSort{
  buildGraph(mse)
}

network.filter{
    _.getClass.getSimpleName == "Variable"
}.foreach{
    weights =>
      val size = weights.size
      val (m,n) = (size._1.asInstanceOf[Int], size._2.asInstanceOf[Int])
      weights.value = Nd4j.randn(m,n) * math.sqrt(3/(m.toDouble + n.toDouble))
}

val x_ = Nd4j.readNumpy("resources/boston_x.csv", ",")
val y_ = Nd4j.readNumpy("resources/boston_y.csv", ",")

// standardize data
val xs_ = x_.subRowVector(x_.mean(0)).divRowVector( x_.std(0))
val ys_ = y_.subRowVector(y_.mean(0)).divRowVector( y_.std(0))

// concatenate data
val data = Nd4j.concat(1, ys_, xs_);

val epochs = 500
val batchSize = 100
val stepsPerEpoch = xrows / batchSize

val sgd = new GradientDescent(network, learningRate = 0.1)

for(epoch <- 0 to epochs) {
  var loss = 0d
  for(j <- 0 until stepsPerEpoch) {

    Nd4j.shuffle(data, 1)

    val feedDict: Map[Node, Any] = Map(
      x -> data.getColumns( (1 to nfeatures):_*).getRows((0 until batchSize):_*),
      y -> data.getColumn(0).getRows((0 until batchSize):_*)
    )

    sgd.optimize(feedDict)

    loss += mse.value(0,0)
  }
  if(epoch % 50 == 0)  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
}
```
