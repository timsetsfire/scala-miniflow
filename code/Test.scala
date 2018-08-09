import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.optimize.GradientDescent
import scala.util.Try
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log}
import org.nd4s.Implicits._
import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}

:load code/Graph.scala

val x = new Input()
x.setName("features")
val y = new Input()
y.setName("labels")
val h1 = Tanh(x, (2, 16))
h1.setName("hidden_layer1")
val h2 = Tanh(h1, (16, 8))
h2.setName("hidden_layer2")
val yhat = Sigmoid(h2, (8, 1))
yhat.setName("prediction")
val bce = new BCE(y, yhat)

val x_ = Nd4j.readNumpy("resources/X.csv", ",")
val y_ = Nd4j.readNumpy("resources/y.csv", ",")
val y2_ = Nd4j.concat(1, y_, y_.mul(-1).add(1));


val network = buildGraph(bce)
val dag = topologicalSort(network)

val xrows = x_.shape.apply(0)
val nfeatures = x_.shape.apply(1)

val xs_ = x_.subRowVector(x_.mean(0)).divRowVector( x_.std(0))
//val ys = y_.subRowVector(y_.mean(0)).divRowVector( y_.std(0))

val data = Nd4j.concat(1, y_, x_);
val epochs = 1000
val batchSize = 600
val stepsPerEpoch = xrows / batchSize


dag.foreach( node =>
  if(node.getClass.getSimpleName == "Variable") {
    val (m,n) = node.size
    node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int])*0.1
  }
)

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

    loss += bce.value(0,0)
  }
  if(epoch % 100 == 0)  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
}



val feedDict: Map[Node, Any] = Map(
x -> data.getColumns( (1 to nfeatures):_*).getRows((0 until 600):_*),
y -> data.getColumn(0).getRows((0 until 600):_*)
)
feedDict.foreach{ n => n._1.value = n._2.asInstanceOf[INDArray]}
dag.foreach( _.forward())
