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

import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log}
// val temp = ((y_ * log(yhat_))) + ((y_.sub(1).mul(-1))*log(yhat_.sub(1).mul(-1)))
// temp.sum(0)


//******//
import breeze.linalg._
import breeze.plot._
// import scala.concurrent.duration.Thread
val f2 = Figure()
//******//

val x_ = Nd4j.readNumpy("resources/digits_x.csv", ",").div(16)

val realImages =  new Input()
realImages.setName("real_images")
val realLabels = new Input()
realLabels.setName("real_labels")

// real and fake data

// generator
val noise = new Input()
noise.setName("noise")
val fakeLabels = new Input()
fakeLabels.setName("fake_labels")
val h1Generator= ReLU(noise, (256, 64))
h1Generator.setName("generator_hidden1")
val fakeImages = ReLU(h1Generator, (64,64))
fakeImages.setName("fake_images")


// discriminator
// values are set depending on which network we are training
val images = new Input()
images.setName("images")
val labels = new Input()
labels.setName("labels")

// discriminator
val h1Discrim = Maxout(images, (64, 1))
h1Discrim.setName("discriminator_hidden_layer1")
val logits = Linear(h1Discrim, (1, 1))
val cost = new BceWithLogits(labels, logits)
cost.setName("discriminator_cost")

// val ganNetwork = buildGraph(cost):Reset
// val ganNetwork = topologicalSort(ganNetwork)

val generatorNetwork = buildGraph(fakeImages)
val generator = topologicalSort(generatorNetwork)

val discriminatorNetwork = buildGraph(cost)
val discriminator = topologicalSort(discriminatorNetwork)

// generate fake fake

// how to move through

// initialize generator and discriminator
generator.foreach( node =>
  if(node.getClass.getSimpleName == "Variable") {
    val (m,n) = node.size
    node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int])*(1/(math.sqrt(m.asInstanceOf[Int].toDouble + n.asInstanceOf[Int].toDouble)))
  }
)

// initialize generator and discriminator
discriminator.foreach( node =>
  if(node.getClass.getSimpleName == "Variable") {
    val (m,n) = node.size
    node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int])*(1/(math.sqrt(m.asInstanceOf[Int].toDouble + n.asInstanceOf[Int].toDouble)))
  }
)
//****************************************

val Array(xrows, xcols) = x_.shape
val epochs = 500
val batchSize = 32
val stepsPerEpoch = xrows / batchSize
val discriminatorTrainables = discriminator.filter{ _.getClass.getSimpleName == "Variable" }
val generatorTrainables = generator.filter{ _.getClass.getSimpleName == "Variable" }



for(epoch <- 0 to epochs) {
  var loss = 0d
  for(steps <- 0 to stepsPerEpoch) {
    //f2.clear
    val noiseData = Nd4j.rand(batchSize,256).mul(2).sub(1)
    val fakeLabelData = Nd4j.ones(batchSize, 1)

    val generatorFeedDict: Map[Node, INDArray] = Map(
      noise -> noiseData,
      fakeLabels -> fakeLabelData
    )

    // generator
    generatorFeedDict.foreach{ case (n, v) => n.forward(v)}
    generator.foreach(_.forward())
    images.forward(fakeImages.value)
    labels.forward(fakeLabels.value)
    discriminator.foreach(_.forward())
    discriminator.reverse.foreach(_.backward())
    //fakeImages.gradients(fakeImages) = images.gradients(images)
    fakeImages.backward(images.gradients(images))
    generator.reverse.tail.foreach(_.backward())
    // still need to update parameters of generator


    val fakeImageData = fakeImages.value
    Nd4j.shuffle(x_,1)
    val realImageData = x_.getRows((0 until batchSize):_*)
    val realLabelData = Nd4j.ones(batchSize, 1)

    val labelData = Nd4j.concat(0, fakeLabelData, realLabelData)



    val imageData = Nd4j.concat(0, fakeImageData, realImageData)

    val discriminatorFeedDict: Map[Node, INDArray] = Map(
      images -> imageData,
      labels -> labelData
    )


    discriminatorFeedDict.foreach{ case (n, v) => n.forward(v)}
    discriminator.foreach(_.forward())
    discriminator.reverse.foreach(_.backward())
    // still need to update parameters for discriminator

    val learningRate = 0.01
    for(t <- discriminatorTrainables) {
      val partial = t.gradients(t)
      t.value.subi(  partial * 0.01 )
    }
    for(t <- generatorTrainables) {
      val partial = t.gradients(t)
      t.value.subi(  partial * 0.001 )
    }

    println("generative grad sum")
    generator.foreach( i => i.gradients.mapValues( v => println(v.sumT)))
    println("discriminator grad sum")
    discriminator.foreach( i => i.gradients.mapValues( v => println(v.sumT)))
    loss += cost.value(0,0)


    //* plot */
    val dig1 = fakeImages.value.getRow(6).data.asDouble()
    val dig1b = DenseMatrix(dig1).reshape(8,8).t
    f2.subplot(0) += image(dig1b)
    // Thread.sleep(5)
    //*******/
  }
  if(epoch % 10 == 0) println(s"loss: ${loss}")

}
// start forward and back prop
