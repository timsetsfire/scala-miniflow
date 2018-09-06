import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp, abs}

import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._

val in = Nd4j.randn(4,3)
val x = new Input()
val lr = new LeakyReLu(x, 0.01)

x.forward(in)
lr.forward()
lr.backward( Nd4j.ones(1,1))
x.backward()


//
// class LeakyReLu(node: Node, l: Double) extends Node(List(node)) {
//
//   override def forward(value: INDArray = null): Unit = {
//     val in = inboundNodes(0).value
//     this.value = (in add in.mul(l)).div(2d) add abs(in sub in.mul(l)).div(2d)
//   }
//
//   override def backward(value: INDArray = null): Unit = {
//     this.inboundNodes.foreach{
//       n =>
//         val Array(rows, cols) = this.value.shape
//         this.gradients(n) = Nd4j.zeros(rows, cols)
//     }
//     if(value == null) {
//       this.outboundNodes.foreach{
//         n =>
//         val gradCost = n.gradients(this)
//         val out = this.value
//         this.gradients(this.inboundNodes(0)) += ((in gt out) add (out gt in))*gradCost
//       }
//     } else {
//       this.gradients(this) = value
//       val gradCost = this.gradients(this)
//       val out = this.value
//       this.gradients(this.inboundNodes(0)) += ((in gt out) add (out gt in))*gradCost
//     }
//   }
// }
