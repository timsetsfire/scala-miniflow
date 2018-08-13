import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.regularization.Dropout
import com.github.timsetsfire.nn.optimize.GradientDescent
import scala.util.Try
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log}
import org.nd4s.Implicits._
import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}

:load code/Graph.scala

import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log}
// val temp = ((y_ * log(yhat_))) + ((y_.sub(1).mul(-1))*log(yhat_.sub(1).mul(-1)))
// temp.sum(0)


def setDropoutTraing(n: Node, training: Boolean = false): Unit = {
  n.asInstanceOf[Dropout[Node]].train = training
}


val x_ = Nd4j.randn(1000, 30)
val y_ = Nd4j.randn(1000, 1)

val x =  new Input()
x.setName("features")
val y = new Input()
y.setName("labels")


val d1 = new Dropout(x, dropout = 0.5)
val yhat = Linear(d1, (30, 1))
val mse = new MSE(y, yhat)

x.forward(x_)
y.forward(y_)

val network = buildGraph(mse)
val dag = topologicalSort(network)
val Array(xrows, xcols) = x_.shape
val epochs = 100
val batchSize = 512
val stepsPerEpoch = xrows / batchSize
val trainables = dag.filter{ _.getClass.getSimpleName == "Variable" }

trainables.foreach{ node =>
    val (m,n) = node.size
    node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int]) * math.sqrt(3/(m.asInstanceOf[Int].toDouble + n.asInstanceOf[Int].toDouble))
}
val learningRate = 0.01
for(i <- 0 to 1000) {
  dag.foreach( _.forward())
  dag.reverse.map{ i => (i, Try(i.backward()))}
  val trainables = dag.filter{ _.getClass.getSimpleName == "Variable" }
  for(t <- trainables) {
    val partial = t.gradients(t)
    t.value.subi(  partial * learningRate )
  }
  if(i % 100 == 0) println(s"loss: ${mse.value}")
}


d1.train = false

dag.foreach( _.forward() )
println(mse.value)

val const = Nd4j.ones( x_.shape.apply(0), 1)
val X_ = Nd4j.concat(1, const, x_)
import org.nd4j.linalg.inverse.InvertMatrix.invert

val beta = invert(X_.transpose.dot(X_), false).dot(X_.transpose).dot(y_)
val yhat = X_.dot(beta)
val e = (yhat - y_)
println("\n")
println((e*e).sum(0).sum(1) / X_.shape.apply(0).toDouble)
println(mse.value)
