import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.activation._
import com.github.timsetsfire.nn.costfunctions._
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





    def buildGraph(terminalNode: Node) = {
      val m = MutMap[Node,ArrayBuffer[Node]]()

      def helper( t: Node ): Unit = {
        if(t.inboundNodes.length == 0) m.update(t, ArrayBuffer())
        else {
          m.update(t, ArrayBuffer(t.inboundNodes:_*))
          t.inboundNodes.map(helper)
        }
      }
      helper(terminalNode)
      m
    }

    def topologicalSort(inputs: MutMap[Node, ArrayBuffer[Node]],
                        sorted: ArrayBuffer[Node]=ArrayBuffer()): ArrayBuffer[Node] = {
      val (inputNodesMap, otherNodesMap) = inputs.partition{ _._2.isEmpty}
      if(inputNodesMap.isEmpty) {
        if(otherNodesMap.isEmpty) sorted else sys.error("graph has at least one cycle")
      } else {
        val inputNodes = inputNodesMap.map{ _._1 }
        val next = MutMap() ++ otherNodesMap.mapValues{ inputs => inputs -- inputNodes }
        topologicalSort( next, sorted ++ inputNodes)
      }
    }
    def setDropoutTraining(n: Node, training: Boolean = false): Unit = {
      n.asInstanceOf[Dropout[Node]].train = training
    }

    val stepSize: Double = 0.001
    val beta1: Double = 0.9
    val beta2: Double = 0.999
    val delta: Double = 1e-8


    val epochs = 10

    val x_ = Nd4j.readNumpy("resources/digits_x.csv", ",").sub(8).div(8)

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
    val h1Generator= LeakyReLU(noise, (5,5), 0.2)
    h1Generator.setName("generator_hidden1")
    val h2Generator= LeakyReLU(h1Generator, (5, 5), 0.2)
    h2Generator.setName("generator_hidden2")
    val h3Generator= LeakyReLU(h2Generator, (5, 5), 0.2)
    h2Generator.setName("generator_hidden3")
    val fakeImages = Tanh(h3Generator, (5,5))
    fakeImages.setName("fake_images")


    // discriminator
    // values are set depending on which network we are training
    val images = new Input()
    images.setName("images")
    val labels = new Input()
    labels.setName("labels")

    // discriminator
    val h1Discrim = LeakyReLU(images, (5,5), 0.2)
    h1Discrim.setName("discriminator_hidden_layer1")
    val d1 = new Dropout(h1Discrim, 0.8)
    d1.setName("dropout_h1_layer")
    val h2Discrim = LeakyReLU(d1, (5,2), 0.2)
    h2Discrim.setName("discriminator_hidden_layer2")
    val d2 = new Dropout(h2Discrim, 0.8)
    d2.setName("dropout_h2_layer")
    val logits = Linear(d2, (2,1))
    logits.setName("discriminator_logits")

    val cost = new BceWithLogits(labels, logits)
    cost.setName("discriminator_cost")

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
    //****************************************
    val Array(xrows, xcols) = x_.shape
    val batchSize = 128
    val stepsPerEpoch = xrows / batchSize

    val firstMomentGenerator = generatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
    val secondMomentGenerator = generatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
    val firstMomentDiscriminator = discriminatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
    val secondMomentDiscriminator = discriminatorTrainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
    val t = new java.util.concurrent.atomic.AtomicInteger

    // //******//
    import breeze.linalg._
    import breeze.plot._
    // import scala.concurrent.duration.Thread
    val f2 = Figure()
    //******//
    val noiseDataForPicture = Nd4j.rand(9,5).mul(2).sub(1)


    // for(epoch <- 0 to epochs) {

      var loss = 0d
      var genCost = 0d
      var n = 0d
      // for(steps <- 0 to stepsPerEpoch) {


        val noiseData = Nd4j.rand(batchSize,5).mul(2).sub(1)
        val fakeLabelData = Nd4j.zeros(batchSize, 1)

        val generatorFeedDict: Map[Node, INDArray] = Map(
          noise -> noiseData,
          fakeLabels -> fakeLabelData
        )

// for(i <- 0 to 10) {
        // generator
        t.addAndGet(1)
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


        // for( n <- generatorTrainables) {
        //   firstMomentGenerator(n).muli(beta1).addi(n.gradients(n).mul(1 - beta1))
        //   secondMomentGenerator(n).muli(beta2).addi( pow(n.gradients(n),2).mul(1 - beta2))
        //   val fhat = firstMomentGenerator(n).div(1 - math.pow(beta1, t.get))
        //   val shat = secondMomentGenerator(n).div(1 - math.pow(beta2, t.get))
        //   n.value.addi( fhat.mul(-stepSize).div(sqrt(shat).add(delta)))
        // }
        // println("\n")
        // generatorTrainables.foreach( i => print( abs(i.gradients(i)).sumT, ","))
        // println("\n")
// }



//         val fakeImageData = fakeImages.value
//         Nd4j.shuffle(x_,1)
//         val realImageData = x_.getRows((0 until batchSize):_*)
//         val realLabelData = Nd4j.ones(batchSize, 1)
//         val fakeLabelData0 = Nd4j.zeros(batchSize, 1)
//
//         val labelData = Nd4j.concat(0, fakeLabelData0, realLabelData)
//         val imageData = Nd4j.concat(0, fakeImageData, realImageData)
//         val discriminatorFeedDict: Map[Node, INDArray] = Map(
//           images -> imageData,
//           labels -> labelData
//         )
//
//         discriminator.filter{ _.getClass.getSimpleName == "Dropout"}.foreach(d => setDropoutTraining(d, true))
//         discriminatorFeedDict.foreach{ case (n, v) => n.forward(v)}
//         discriminator.foreach(_.forward())
//         discriminator.reverse.foreach(_.backward())
//
//         for( n <- discriminatorTrainables) {
//           firstMomentDiscriminator(n).muli(beta1).addi(n.gradients(n).mul(1 - beta1))
//           secondMomentDiscriminator(n).muli(beta2).addi( pow(n.gradients(n),2).mul(1 - beta2))
//           val fhat = firstMomentDiscriminator(n).div(1 - math.pow(beta1, t.get))
//           val shat = secondMomentDiscriminator(n).div(1 - math.pow(beta2, t.get))
//           n.value.addi( fhat.mul(-stepSize).div(sqrt(shat).add(delta)))
//         }
//
//
//         // // still need to update parameters for discriminator
//         // val learningRate = 0.001
//         // for(t <- discriminatorTrainables) {
//         //   val partial = t.gradients(t)
//         //   t.value.subi(  partial * 0.001 )
//         // }
//         // for(t <- generatorTrainables) {
//         //   val partial = t.gradients(t)
//         //   t.value.subi(  partial * 0.001 )
//         // }
//         //decay /= 2d
//         loss += ((cost.value(0,0)) * images.value.shape.apply(0))
//         n += images.value.shape.apply(0)
//       }
//
//       if(epoch % 100 == 0) {
//         println(s"discriminator -> epoch: ${epoch}, loss: ${loss / n.toDouble}")
//         println(s"generator -----> epoch: ${epoch}, loss: ${genCost / (n.toDouble/2d)}")
//       }
//
//       // check
//       // fakeImages.gradients(fakeImages).sumT = 0.004577898420393467
//
//       }
//       noise.forward(noiseDataForPicture)
//       generator.foreach(_.forward())
//       f2.clear()
//       for (i <- 0 until 9) {
//         val dig1 = fakeImages.value.getRow(i).dup.data.asDouble()
//         val dig1b = DenseMatrix(dig1).reshape(8,8).t
//         f2.subplot(3,3,i) += image(dig1b)
//       }
//
//
//
//
//
// //* plot */
//
// //*******/
// // }
