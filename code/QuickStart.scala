import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log,pow,sqrt}
import org.nd4s.Implicits._

import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.regularization._
import com.github.timsetsfire.nn.optimize._
import com.github.timsetsfire.nn.graph._

val x = new Input()
val y = new Input()
val h1 = ReLU(x, (13,6))
val yhat = Linear(h1, (6,1))
val mse = new MSE(y, yhat)
val network = topologicalSort{
  buildGraph(mse)
}

network.filter{
    _.getClass.getSimpleName == "Variable"
}.foreach{
    weights =>
      val size = weights.size
      val (m,n) = (size._1.asInstanceOf[Int], size._2.asInstanceOf[Int])
      weights.value = Nd4j.randn(m,n) * math.sqrt(3/(m.toDouble + n.toDouble))
}

val x_ = Nd4j.readNumpy("resources/boston_x.csv", ",")
val y_ = Nd4j.readNumpy("resources/boston_y.csv", ",")

// standardize data
val xs_ = x_.subRowVector(x_.mean(0)).divRowVector( x_.std(0))
val ys_ = y_.subRowVector(y_.mean(0)).divRowVector( y_.std(0))

// concatenate data
val data = Nd4j.concat(1, ys_, xs_);

val epochs = 500
val batchSize = 100
val stepsPerEpoch = xrows / batchSize

val sgd = new GradientDescent(network, learningRate = 0.1)

for(epoch <- 0 to epochs) {
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
  if(epoch % 50 == 0)  println(s"Epoch: ${epoch}, Loss: ${loss/stepsPerEpoch.toDouble}")
}
