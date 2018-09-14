import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.graph._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
import com.github.timsetsfire.nn.batchnormalization._
import com.github.timsetsfire.nn.regularization.Dropout
import com.github.timsetsfire.nn.optimize._
import scala.util.Try
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid,exp,log}
import org.nd4s.Implicits._
import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}

import org.nd4j.linalg.ops.transforms.Transforms._
// val temp = ((y_ * log(yhat_))) + ((y_.sub(1).mul(-1))*log(yhat_.sub(1).mul(-1)))
// temp.sum(0)
import java.io.{FileInputStream, BufferedInputStream, File}


object Gan extends App {


    def setDropoutTraining(n: Node, training: Boolean = false): Unit = {
      n.asInstanceOf[Dropout[Node]].train = training
    }

    // //******//
    import breeze.linalg._
    import breeze.plot._
    // import scala.concurrent.duration.Thread
    // val f2 = Figure()
    //******//
    val noiseDataForPicture = Nd4j.rand(16,100).mul(2).sub(1)

    var stepSize: Double = 0.002 // 0.001 default
    val beta1: Double = 0.2  // 0.9 default
    val beta2: Double = 0.999  // 0.999 default
    val delta: Double = 1e-8


    val epochs = args(0).toInt
    // val epochs = 500
    // val x_ = Nd4j.readNumpy("resources/digits_x.csv", ",").sub(8).div(8)
    val x_ = Nd4j.readNumpy("resources/mnist_test.csv", ",").getColumns( (1 until 785):_*).div(255.0)

    // data placeholders
    val realImages =  new Input()
    realImages.setName("real_images")
    val realLabels = new Input()
    realLabels.setName("real_labels")

    // generator
    val noise = new Input()
    noise.setName("noise")
    val fakeLabels = new Input()
    fakeLabels.setName("fake_labels")
    val h1Generator= LeakyReLU(noise, (100,128), 0.2)
    h1Generator.setName("generator_hidden1")
    // val bn1 = BatchNormalization(h1Generator, (None, 128))
    // val h2Generator= LeakyReLU(bn1, (128, 256), 0.2)
    val h2Generator= LeakyReLU(h1Generator, (128, 256), 0.1)
    h2Generator.setName("generator_hidden2")
    // val bn2 = BatchNormalization(h2Generator, (None, 256))
    // val h3Generator= LeakyReLU(bn2, (256, 512), 0.2)
    val h3Generator= LeakyReLU(h2Generator, (256, 512), 0.1)
    h3Generator.setName("generator_hidden3")
    // val bn3 = BatchNormalization(h3Generator, (None, 512))
    val fakeImages = Sigmoid(h3Generator, (512,784))
    fakeImages.setName("fake_images")
	  // end generator


    // discriminator
    // values are set depending on which network we are training
    val images = new Input()
    images.setName("images")
    val labels = new Input()
    labels.setName("labels")

    // discriminator

	  val h1Discrim = LeakyReLU(images, (784,256), 0.1)
	//val h1Discrim = Maxout(images, (784,256))
    h1Discrim.setName("discriminator_hidden_layer1")

    val d1 = new Dropout(h1Discrim, 0.01)
    d1.setName("dropout_h1_layer")

    // val bn1Discrim = BatchNormalization(d1, (None, 256))
    // val h2Discrim = LeakyReLU(bn1Discrim, (256,64), 0.1)
    val h2Discrim = LeakyReLU(d1, (256,64), 0.1)
	//val h2Discrim = Maxout(d1, (256,64))
    h2Discrim.setName("discriminator_hidden_layer2")
    val d2 = new Dropout(h2Discrim,0.01)
    d2.setName("dropout_h2_layer")

    // val bn2Discrim = BatchNormalization(d2, (None, 64))
    // val h3Discrim = LeakyReLU(bn2Discrim, (64,16), 0.1)
    val h3Discrim = LeakyReLU(d2, (64,16), 0.1)
	//val h3Discrim = Maxout(d2, (64,16))

    val logits = Linear(h3Discrim, (16, 1))
    logits.setName("discriminator_logits")

    val cost = new BceWithLogits(labels, logits)
    cost.setName("discriminator_cost")

	// end discriminator


  // // generator - from paper
  // val noise = new Input()
  // noise.setName("noise")
  // val fakeLabels = new Input()
  // fakeLabels.setName("fake_labels")
  // val h1Generator= ReLU(noise, (100,1200))
  // h1Generator.setName("generator_hidden1")
  // // val bn1 = BatchNormalization(h1Generator, (None, 128))
  // // val h2Generator= LeakyReLU(bn1, (128, 256), 0.2)
  // val h2Generator= ReLU(h1Generator, (1200, 1200))
  // h2Generator.setName("generator_hidden2")
  // // val bn2 = BatchNormalization(h2Generator, (None, 256))
  // // val h3Generator= LeakyReLU(bn2, (256, 512), 0.2)
  // // val bn3 = BatchNormalization(h3Generator, (None, 512))
  // val fakeImages = Sigmoid(h2Generator, (1200,784))
  // fakeImages.setName("fake_images")
  // // end generator
  //
  // // discriminator - from paper
  // // values are set depending on which network we are training
  // val images = new Input()
  // images.setName("images")
  // val labels = new Input()
  // labels.setName("labels")
  // val h1Discrim = Maxout(images, (784,240))
  // h1Discrim.setName("discriminator_hidden_layer1")
  // val h2Discrim = Maxout(h1Discrim, (240,240))
  // h2Discrim.setName("discriminator_hidden_layer2")
  // val logits = Linear(h2Discrim, (240, 1))
  // logits.setName("discriminator_logits")

  // val cost = new BceWithLogits(labels, logits)
  // cost.setName("discriminator_cost")


  val generatorNetwork = buildGraph(fakeImages)
  val generator = topologicalSort(generatorNetwork)

  val discriminatorNetwork = buildGraph(cost)
  val discriminator = topologicalSort(discriminatorNetwork)

  val discriminatorTrainables = discriminator.filter{ _.getClass.getSimpleName == "Variable" }
  val generatorTrainables = generator.filter{ _.getClass.getSimpleName == "Variable" }


  // initialize generator and discriminator
  discriminatorTrainables.foreach{ node =>
      val (m,n) = node.size
      node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int]) * math.sqrt(3/(m.asInstanceOf[Int].toDouble + n.asInstanceOf[Int].toDouble))
  }

  // initialize generator and discriminator
  generatorTrainables.foreach{ node =>
      val (m,n) = node.size
      node.value = Nd4j.randn(m.asInstanceOf[Int], n.asInstanceOf[Int]) * math.sqrt(3/(m.asInstanceOf[Int].toDouble + n.asInstanceOf[Int].toDouble))
  }


    // val Seq(s1,g1,b1) = bn1.inboundNodes
    // val Seq(s2,g2,b2) = bn2.inboundNodes
    // val Seq(s3,g3,b3) = bn3.inboundNodes

    // g1.value = Nd4j.rand( g1.size._1.asInstanceOf[Int], g1.size._2.asInstanceOf[Int])
    // g2.value = Nd4j.rand( g2.size._1.asInstanceOf[Int], g2.size._2.asInstanceOf[Int])
    // g3.value = Nd4j.rand( g3.size._1.asInstanceOf[Int], g3.size._2.asInstanceOf[Int])


  //****************************************
  val Array(xrows, xcols) = x_.shape
  val batchSize = 128
  val stepsPerEpoch = xrows / batchSize

  val firstMomentGenerator = generatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val secondMomentGenerator = generatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val firstMomentDiscriminator = discriminatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val secondMomentDiscriminator = discriminatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val t = new java.util.concurrent.atomic.AtomicInteger

  for(epoch <- 0 to epochs) {

    var loss = 0d
    var genCost = 0d
    var n = 0d
    for(steps <- 0 to stepsPerEpoch) {

      t.addAndGet(1)

      val noiseData = Nd4j.rand(batchSize,100).mul(2).sub(1)
      val fakeLabelData = Nd4j.ones(batchSize, 1)

      val generatorFeedDict: Map[Node, INDArray] = Map(
        noise -> noiseData,
        fakeLabels -> fakeLabelData
      )

      // generator
      discriminator.filter{ _.getClass.getSimpleName == "Dropout"}.foreach(d => setDropoutTraining(d, false))
      generatorFeedDict.foreach{ case (n, v) => n.forward(v)}
      generator.foreach(_.forward())
      images.forward(fakeImages.value)
      labels.forward(fakeLabels.value)
      discriminator.foreach(_.forward())
      discriminator.reverse.foreach(_.backward())
      //fakeImages.gradients(fakeImages) = images.gradients(images)
      fakeImages.backward(images.gradients(images).dup)
      generator.reverse.tail.foreach(_.backward())
      // still need to update parameters of generator
      genCost += (cost.value.sumT*batchSize)

      for( n <- generatorTrainables) {
        firstMomentGenerator(n).muli(beta1).addi(n.gradients(n).mul(1 - beta1))
        secondMomentGenerator(n).muli(beta2).addi( pow(n.gradients(n),2).mul(1 - beta2))
        val fhat = firstMomentGenerator(n).div(1 - math.pow(beta1, t.get))
        val shat = secondMomentGenerator(n).div(1 - math.pow(beta2, t.get))
        n.value.addi( fhat.mul(-stepSize).div(sqrt(shat).add(delta)))
      }


      generator.foreach(_.forward())
      val fakeImageData = fakeImages.value
      Nd4j.shuffle(x_,1)
      val realImageData = x_.getRows((0 until batchSize):_*)
      val realLabelData = Nd4j.ones(batchSize, 1)
      val fakeLabelData0 = Nd4j.zeros(batchSize, 1)

      val labelData = Nd4j.concat(0, fakeLabelData0, realLabelData)
      val imageData = Nd4j.concat(0, fakeImageData, realImageData)
      val discriminatorFeedDict: Map[Node, INDArray] = Map(
        images -> imageData,
        labels -> labelData
      )

      discriminator.filter{ _.getClass.getSimpleName == "Dropout"}.foreach(d => setDropoutTraining(d, true))
      discriminatorFeedDict.foreach{ case (n, v) => n.forward(v)}
      discriminator.foreach(_.forward())
      discriminator.reverse.foreach(_.backward())

      for( n <- discriminatorTrainables) {
        firstMomentDiscriminator(n).muli(beta1).addi(n.gradients(n).mul(1d - beta1))
        secondMomentDiscriminator(n).muli(beta2).addi( pow(n.gradients(n),2).mul(1d - beta2))
        val fhat = firstMomentDiscriminator(n).div(1 - math.pow(beta1, t.get))
        val shat = secondMomentDiscriminator(n).div(1 - math.pow(beta2, t.get))
        n.value.addi( fhat.mul(-stepSize).div(sqrt(shat).add(delta)))
      }

      loss += ((cost.value(0,0)) * images.value.shape.apply(0))
      n += images.value.shape.apply(0)
    }
    // if(epoch % 1000 == 0) stepSize /= 2d
    if(epoch % 10 == 0) {
      println(s"discriminator -> epoch: ${epoch}, loss: ${loss / n.toDouble}")
      println(s"generator -----> epoch: ${epoch}, loss: ${genCost / (n.toDouble/2d)}")
      noise.forward(noiseDataForPicture)
      generator.foreach(_.forward())
      val f4 = Figure()
      for (i <- 0 until 16) {
        val dig1 = fakeImages.value.getRow(i).dup.data.asDouble()
        val da = (for{ i <- 0 to 28} yield dig1.drop(i*28).take(28)).init
        val dig1b = DenseMatrix(da.reverse:_*) //.reshape(28,28)
        f4.subplot(4,4,i) += image(dig1b)
      }
      f4.saveas(s"resources/genfig${epoch}.png")
    }

    // check
    // fakeImages.gradients(fakeImages).sumT = 0.004577898420393467

    // if(i % 500)
    }

  }
