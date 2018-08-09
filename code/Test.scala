import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._

:load code/Graph.scala

val x = new Input()
x.setName("features")
val y = new Input()
y.setName("label")
val h1 = Tanh(x, (2, 16))
h1.setName("hidden_layer1")
val h2 = Tanh(h1, (16, 8))
h2.setName("hidden_layer2")
val yhat = Sigmoid(h2, (8, 1))
yhat.setName("prediction")
val ce = new BCE(y, yhat)

import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}


// def buildGraph(terminalNode: Node) = {
//   val m = MutMap[Node,ArrayBuffer[Node]]()
//
//   def helper( t: Node ): Unit = {
//     if(t.inboundNodes.length == 0) m.update(t, ArrayBuffer())
//     else {
//       m.update(t, ArrayBuffer(t.inboundNodes:_*))
//       t.inboundNodes.map(helper)
//     }
//   }
//   helper(terminalNode)
//   m
// }

val network = buildGraph(ce)
topologicalSort(network)
