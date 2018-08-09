

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp}
import com.github.timsetsfire.nn.node._

implicit val graph = MutMap[Node, ArrayBuffer[Node]]()

abstract class CostFunction(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(y,yhat))(graph) {
  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit
}

object Graph {
  def buildGraph[T <: CostFunction](cf: T):  MutMap[Node, ArrayBuffer[Node]] = MutMap()
  def buildDag[T <: CostFunction](cf: T): ArrayBuffer[Node] = ArrayBuffer()
}
