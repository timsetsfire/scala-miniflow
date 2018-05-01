# scala-miniflow

Came across this during my Udacity Deep Learning Nanodegree and thought it was very cool.
The original library, MiniFlow, was a neural network library written from scratch (in Python)
that behaves much like TensorFlow, Google's deep learning library.

This started out as a port of MiniFlow to Scala, but I made some minor changes
along the way.  

See
(https://medium.com/udacity/the-miniflow-lesson-929200f72e27) and (https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd)
for more information.  

## Usage

Suppose that `x_` and `y_` are INDArray of data correspond to target and features respectively.  

Standardize Data

```
import com.github.timsetsfire.nn.node._
import scala.collection.mutable.{Map => MutMap, ArrayBuffer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

// standardize data
//x_.subiRowVector(x_.mean(0)).diviRowVector(x_.std(0))
//y_.subiRowVector(y_.mean(0)).diviRowVector(y_.std(0))

// initialize graph
implicit val graph = MutMap[Node, ArrayBuffer[Node]]()
val x = new Input()
val y = new Input()
// trainables
val b = new Variable() // be good to initialize
val w = new Variable() // be good to initialize
// output
val yhat = (x * w) + b
// or
// val yhat = Linear(x,w,b)
val mse = MSE(y, yhat)
```
