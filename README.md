# scala-miniflow

Came across this during my Udacity Deep Learning Nanodegree and thought it was very cool.
The original library, MiniFlow, was a neural network library written from scratch (in Python)
that behaves much like TensorFlow, Google's deep learning library.

This started out as a port of MiniFlow to Scala, but I made some minor changes
along the way.  

Currently redoing topological sort and working on implementing Adam optimizer.  

See
(https://medium.com/udacity/the-miniflow-lesson-929200f72e27) and (https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd)
for more information.  

## Usage

Suppose that `x_` and `y_` are INDArray of data correspond to target and features respectively.  


```
import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.graph.topologicalSort
import com.github.timsetsfire.nn.optimize.GradientDescent
import scala.collection.mutable.{Map => MutMap, ArrayBuffer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

// standardize data
//x_.subiRowVector(x_.mean(0)).diviRowVector(x_.std(0))
//y_.subiRowVector(y_.mean(0)).diviRowVector(y_.std(0))

// initialize graph
// each key is a node and each value is the set of nodes feeding the key

implicit val graph = MutMap[Node, ArrayBuffer[Node]]()

val x = new Input()
val y = new Input()
val h1 = ReLU(x, (13, 10), "xavier")
val h2 = ReLU(h1, (10, 10), "xavier")
val yhat = Linear(h2, (10, 1))
val mse = MSE(y,yhat)

// sort graph
val dag = topologicalSort(graph)

// set some details
val xrows = x_.shape.apply(0)
val nfeatures = x_.shape.apply(1)
val data = Nd4j.concat(1, y_, x_);
val epochs = 100
val batchSize = 32
val stepsPerEpoch = xrows / batchSize

// initialize parameters
dag.foreach( n => if(n.getClass.getName.endsWith("Variable")) initializeParams(n.asInstanceOf[Variable]))

val sgd = new GradientDescent(dag, learningRate = 0.1)
for(epoch <- 0 until epochs) {
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
  if(epoch % 5 == 0)  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
}

```
