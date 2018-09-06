
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{pow,normalizeZeroMeanAndUnitVariance=>stdize}
import org.nd4j.linalg.inverse.InvertMatrix.invert

def r2(y: INDArray, yhat: INDArray) = {
  val rss = pow(y sub yhat,2)
  val tss = pow(y.subRowVector(y.mean(0)),2)
  1d - rss.sumT / tss.sumT
}
def cost(y: INDArray, x: INDArray, b: INDArray) = {
 (y - x.mmul(b)).norm2T / y.shape.apply(0).toDouble
}

object Regression {
  // read in data
  def run() = {
    val x = Nd4j.readNumpy("resources/boston_x.csv", ",")
    val y = Nd4j.readNumpy("resources/boston_y.csv", ",")

    // standardize data.
    // using the imported stdize will standardize inplace
    val xs = x.subRowVector(x.mean(0)).divRowVector(x.std(0))
    val ys = y.subRowVector(y.mean(0)).divRowVector(y.std(0))

    // estimate weight matrix b
    val b = invert(xs.transpose mmul xs, false) mmul xs.transpose mmul ys

    val yhat = xs mmul b

    println( f"r^2: ${r2(ys,yhat)}%2.3f")
  }
}

object SgdRegression  {

  def run() = {

    // read in data
    val x = Nd4j.readNumpy("resources/boston_x.csv", ",")
    val y = Nd4j.readNumpy("resources/boston_y.csv", ",")

    // standardize data.  
    // using the imported stdize will standardize inplace
    val xs = x.subRowVector(x.mean(0)).divRowVector(x.std(0))
    val ys = y.subRowVector(y.mean(0)).divRowVector(y.std(0))

    // initialize weights
    val b = Nd4j.randn(xs.shape.apply(1),1)

    val Array(xrows, xcols) = xs.shape
    val batchSize = 128
    val stepsPerEpoch = xrows / batchSize
    val epochs = 500
    val t = new java.util.concurrent.atomic.AtomicInteger
    val data = Nd4j.concat(1, ys, xs)
    val learningRate = 0.01

    for(epoch <- 0 to epochs) {

      var loss = 0d
      var costValue = 0d
      var n = 0d

      for(steps <- 0 to stepsPerEpoch) {

        t.addAndGet(1)
        Nd4j.shuffle(data,1)
        val xBatch = data( 0 until batchSize, 1 until data.shape.apply(1))
        val yBatch = data( 0 until batchSize, 0)
        val yhat = xBatch.mmul(b)
        val grad = xBatch.transpose.mmul(yBatch sub yhat).div(batchSize).mul(-1)
        b.subi(grad.mul(learningRate))
        loss += cost(yBatch, xBatch, b) * xBatch.shape.apply(0)
        n += xBatch.shape.apply(0)

      }

      if(epoch % 100 == 0) println(f"cost: ${loss/n}%2.3f")
    }

    val yhat = xs.mmul(b)
    println(f"r^2: ${r2(ys, yhat)}%2.3f")
    }
  }

println("regression")
Regression.run()
println("sgd regression")
SgdRegression.run()


