


import scala.collection.mutable.{ArrayBuffer, Map => MutMap, Set => MutSet}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
import org.nd4s.Implicits._

import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.graph.topologicalSort
import com.github.timsetsfire.nn.optimize.GradientDescent
import scala.util.Try
import org.nd4j.linalg.ops.transforms.Transforms.{exp,log}

implicit val graph = MutMap[Node, ArrayBuffer[Node]]()



import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log}
// val temp = ((y_ * log(yhat_))) + ((y_.sub(1).mul(-1))*log(yhat_.sub(1).mul(-1)))
// temp.sum(0)

val x_ = Nd4j.readNumpy("resources/X.csv", ",")
val y_ = Nd4j.readNumpy("resources/y.csv", ",")
//
// val y = new Input()
// y.value = y_
// val yhat = new Input()
// yhat.value = Nd4j.onesLike(y_).muli(0.5)
// val bce = new BCE(y,yhat)
// val mse = new MSE(y,yhat)


//object NN {
 val y2_ = Nd4j.concat(1, y_, y_.mul(-1).add(1));

  val x = new Input()
  val y = new Input()
  val h1 = Tanh(x, (2, 16), "xavier")
  val h2 = Tanh(h1, (16, 8), "xavier")
  val yhat = Linear(h2, (8, 2), "xavier")
  val mse = new CrossEntropyWithLogits(y,yhat)


  val dag = topologicalSort(graph)
  val learningRate = 0.01


  val xrows = x_.shape.apply(0)
  val nfeatures = x_.shape.apply(1)

  val xs_ = x_.subRowVector(x_.mean(0)).divRowVector( x_.std(0))
  //val ys = y_.subRowVector(y_.mean(0)).divRowVector( y_.std(0))

  val data = Nd4j.concat(1, y2_, x_);
  val epochs = 500
  val batchSize = 600
  val stepsPerEpoch = xrows / batchSize


  dag.foreach( node =>
  	if(node.getClass.getName.endsWith("Variable")) {
  		val (m,n) = node.size
  		node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int])*0.1
  	}
  )
  //
  Nd4j.shuffle(data, 1)

  val feedDict: Map[Node, Any] = Map(
    x -> data.getColumns( (2 to nfeatures + 1):_*).getRows((0 until 600):_*),
    y -> data.getColumns( (0 to 1):_*).getRows((0 until 600):_*)
  )
  feedDict.foreach{ n => n._1.value = n._2.asInstanceOf[INDArray]}

 for(i <- 0 to 5000) {
  dag.foreach( _.forward())
  dag.reverse.map{ i => (i, Try(i.backward()))}
  val trainables = dag.filter{ _.getClass.getName.endsWith("Variable")}
  for(t <- trainables) {
    val partial = t.gradients(t)
    t.value.subi(  partial * learningRate )
  }
if(i % 100 == 0) println(s"loss: ${mse.value}")
}

val p = exp(yhat.value)
p.diviColumnVector(p.sum(1))
val yhat_ = Nd4j.argMax(p)
Nd4j.argMax(y2_, 1) eq yhat_

/** dag.foreach( node =>
if(node.getClass.getName.endsWith("Variable")) {
println(node.getClass.getName)
println("\n")
println(node.value)
println("\n"):reset
}
)*/
  //
  // val sgd = new GradientDescent(dag, learningRate = 0.1)
  // for(epoch <- 0 until epochs) {
  //   var loss = 0d
  //   for(j <- 0 until stepsPerEpoch) {
  //
  //     Nd4j.shuffle(data, 1)
  //
  //     val feedDict: Map[Node, Any] = Map(
  //       x -> data.getColumns( (2 to nfeatures):_*).getRows((0 until batchSize):_*),
  //       y -> data.getColumns( (0 to 1):_*).getRows((0 until batchSize):_*)
  //     )
  //
  //     sgd.optimize(feedDict)
  //
  //     loss += mse.value(0,0)
  //   }
  //   if(epoch % 100 == 0)  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
  // }
  //

//
// val feedDict: Map[Node, Any] = Map(
//   x -> data.getColumns( (1 to nfeatures):_*).getRows((0 until 600):_*),
//   y -> data.getColumns( (0 to 1):_*).getRows((0 until 600):_*)
// )
// feedDict.foreach{ n => n._1.value = n._2.asInstanceOf[INDArray]}
// dag.foreach( _.forward())
//

  /**
  def train(args: String = "tanh") = {
    val x = Input()
    val y = Input()
    val b1 = Input()
    val w1 = Input()
    // x.forward( Nd4j.randn(10,5))
    // w1.forward( Nd4j.randn(5,3))
    // b1.forward( Nd4j.randn(1,3))
    val b2 = Input()
    val w2 = Input()


    val l1 = Linear(x,w1,b1)
    val s1 = if(args == "sigmoid") {
      Sigmoid(l1)
    } else if (args == "relu") {
      Relu(l1)
    } else Tanh(l1)
    val l2 = Linear(s1,w2,b2)
    val mse = new MSE(y, l2)

    val Array(xrows, xCols) = x_.shape

    val feedDict: Map[Node, Any] = Map(
      x -> x_,
      w1 -> Nd4j.randn(xCols,10),
      b1 -> Nd4j.randn(1,10),
      w2 -> Nd4j.randn(10,1),
      b2 -> Nd4j.randn(1,1),
      y -> y_
    )

    val nfeatures = x_.shape.apply(1)

    val data = Nd4j.concat(1, y_, x_);

    import TopologicalSort._
    val graph = topologicalSort(feedDict)
    val epochs = 1001
    val m = 32
    val batchSize = m
    val stepsPerEpoch = m / batchSize
    //
    val trainables = List(w1, b1, w2, b2)
  for(epoch <- 0 until epochs) {
    var loss = 0d
    for(j <- 0 until stepsPerEpoch) {
      Nd4j.shuffle(data, 1)
      val xBatch = data.getColumns( (1 to nfeatures):_*).getRows((0 until batchSize):_*)
      val yBatch = data.getColumn(0).getRows((0 until batchSize):_*)
      x.value = xBatch
      y.value = yBatch
      graph.foreach( _.forward())
      graph.reverse.foreach( _.backward() )
      for(t <- trainables) {
        val partial = t.gradients(t)
        t.value.subi(  partial * 0.01)
      }

      loss += graph.last.value(0,0)
    }
    if(epoch % 100 == 0) println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")

  }
  graph
  }
}

// case class Add(x: Node, y: Node) extends Node()
//   def forward() = {
//     this.value = x.value + y.value
//   }
// }
//
// case class MatMul(x: INDArray, y: INDArray) extends Node() {
//
//   def forward() = {
//     this.value = x mmul y
//   }
// }
*/
