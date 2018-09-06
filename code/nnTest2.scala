import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.regularization._
import com.github.timsetsfire.nn.optimize._
import scala.util.Try
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log}
import org.nd4s.Implicits._
import scala.collection.mutable.{ArrayBuffer, Map=>MutMap, Set=>MutSet}

implicit val graph = MutMap[Node, ArrayBuffer[Node]]()

:load code/Graph.scala

import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log}
// val temp = ((y_ * log(yhat_))) + ((y_.sub(1).mul(-1))*log(yhat_.sub(1).mul(-1)))
// temp.sum(0)


if(arg(1) = "mnist") {

val x_ = Nd4j.readNumpy("resources/digits_x.csv", ",")
val y_ = Nd4j.readNumpy("resources/digits_y.csv", ",")


def setDropoutTraining(n: Node, training: Boolean = false): Unit = {
  n.asInstanceOf[Dropout[Node]].train = training
}

import org.nd4j.linalg.indexing.NDArrayIndex;

// ohe for y_
val ypos = y_.data.asInt zipWithIndex
val y2_ = Nd4j.zeros(y_.shape.apply(0), 10)
ypos.foreach{ case (k,v) => y2_.putScalar(v,k,1.0)}





  val x = new Input()
  x.setName("features")
  val y = new Input()
  y.setName("labels")
  val d1 = new Dropout(x, 0.01)
  val h1 = ReLU(d1, (64, 16))
  val d2 = new Dropout(h1, 0.01)
  val h2 = ReLU(d2, (16, 8))
  // h2.setName("hidden_layer2")
  val yhat = Linear(h2, (8, 10))
  // yhat.setName("logits")
  val ce = new CrossEntropyWithLogits(y,yhat)
  // ce.setName("cost")

  val network = buildGraph(ce)
  val dag = topologicalSort(network)

  val learningRate = 0.01

  val xrows = x_.shape.apply(0)
  val nfeatures = x_.shape.apply(1)

  val xs_ = x_.sub(8).div(8)
  //val ys = y_.subRowVector(y_.mean(0)).divRowVector( y_.std(0))

  val data = Nd4j.concat(1, y2_, xs_);
  val epochs = 1000
  val batchSize = 600
  val stepsPerEpoch = xrows / batchSize


  dag.foreach{
      node =>
      if(node.getClass.getSimpleName == "Variable" ) {
  		    val (m,n) = node.size
  		    node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int]).div( math.sqrt(m.asInstanceOf[Int]) + math.sqrt(n.asInstanceOf[Int] ))
  	  }
  }
  //

 dag.filter( _.getClass.getSimpleName == "Dropout").map{ i => setDropoutTraining(i, true) }

 val adam = new Adam(dag)
 var cost = 0d
 var n = 0d
 for(i <- 0 to epochs) {

   var loss = 0d
   for(j <- 0 until stepsPerEpoch) {

    Nd4j.shuffle(data, 1)

      val feedDict: Map[Node, Any] = Map(
        x -> data.getColumns( (9 to 9 + 63):_*).getRows( (0 until batchSize):_*),
        y -> data.getColumns( (0 to 9):_*).getRows( (0 until batchSize):_*)
      )

      adam.optimize(feedDict)
      val n1 = x.value.shape.apply(0)
      val c1 = ce.value.sumT * n1
      cost += c1
      n += n1
   }
  if(i % 100 == 0) println(s"epoch: ${i}, loss: ${cost / n.toDouble}")
}

dag.filter( _.getClass.getSimpleName == "Dropout").map{ i => setDropoutTraining(i, false) }

val feedDict: Map[Node, INDArray] = Map( x -> data.getColumns( (9 to 9 + 63):_*), y -> data.getColumns( (0 to 9):_*))
feedDict.foreach{ case (node, value) => node.forward(value)}
dag.foreach( _.forward())
val p = exp(yhat.value)
p.diviColumnVector(p.sum(1))
val yhat_ = Nd4j.argMax(p,1)
println(s"accuracy ${(Nd4j.argMax(y.value, 1) eq yhat_).sum(0) / y_.shape.apply(0).toDouble}")
}
