


import scala.collection.mutable.{ArrayBuffer, Map => MutMap, Set => MutSet}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.sigmoid
import org.nd4s.Implicits._

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



val feedDict: Map[Node, Any] = Map(
  x -> Nd4j.randn(10,5),
  w1 -> Nd4j.randn(5,5),
  b1 -> Nd4j.randn(1,5),
  w2 -> Nd4j.randn(5,1),
  b2 -> Nd4j.randn(1,1),
  y -> Nd4j.randn(10,1)
)

import TopologicalSort._
val graph = topologicalSort(feedDict)

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
