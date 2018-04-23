


import scala.collection.mutable.{ArrayBuffer, Map => MutMap, Set => MutSet}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
import org.nd4s.Implicits._

val x_ = Nd4j.readNumpy("c:/users/whittakert/desktop/nd4s/resources/x.csv", ",")
val y_ = Nd4j.readNumpy("c:/users/whittakert/desktop/nd4s/resources/y.csv", ",")


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
val s1 = Sigmoid(l1)
val l2 = Linear(s1,w2,b2)
val mse = new MSE(y, l2)

val Array(xrows, xCols) = x_.shape

val feedDict: Map[Node, Any] = Map(
  x -> x_,
  w1 -> Nd4j.randn(xCols,5),
  b1 -> Nd4j.randn(1,5),
  w2 -> Nd4j.randn(5,1),
  b2 -> Nd4j.randn(1,1),
  y -> y_
)

val nfeatures = x_.shape.apply(1)

val data = Nd4j.concat(1, y_, x_);

import TopologicalSort._
val graph = topologicalSort(feedDict)
val epochs = 100
val m = x_.shape.apply(0)
val batchSize = m
val stepsPerEpoch = m / batchSize
//
 val trainables = List(w1, b1, w2, b2)
//
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
      t.value.subi(  partial * 0.1)
    }

    loss += graph.last.value(0,0)
  }
  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
}

// case class Add(x: Node, y: Node) extends Node() {
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
