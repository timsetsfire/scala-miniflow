
package com.github.timsetsfire.nn.optimize

import scala.collection.mutable.ArrayBuffer
import com.github.timsetsfire.nn.node._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._



class GradientDescent(graph: ArrayBuffer[Node], learningRate: Double = 0.01, beta1: Double = 0.9, beta2: Double = 0.999) {

  val trainables = graph.filter{ _.getClass.getName.endsWith("Variable")}

  // TODO implementing Adam
  // val r = trainables.map{ n => (n, Nd4j.zerosLike(n.value))}.toMap
  // val g = trainables.map{ n => (n, Nd4j.zerosLike(n.value))}.toMap

  def optimize(feedDict: Map[Node, Any]) = {
    feedDict.foreach{ n => n._1.value = n._2.asInstanceOf[INDArray]}
    graph.foreach( _.forward())
    graph.reverse.foreach( _.backward() )
    for(t <- trainables) {
      val partial = t.gradients(t)
      t.value.subi(  partial * learningRate )
    }
  }
}

class Adam(graph: ArrayBuffer[Node], stepSize: Double = 0.001,
  beta1: Double = 0.9, beta2: Double = 0.999, delta: Double = 1e-8
) {
  val trainables = graph.filter{ _.getClass.getSimpleName == "Variable"}
  // add a and train = true
  val firstMoment = trainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val secondMoment = trainables.map{ i => (i, Nd4j.zerosLike(i.value))}.toMap
  val t = new java.util.concurrent.atomic.AtomicInteger

  def optimize(feedDict: Map[Node, Any]) = {
    feedDict.foreach{ n => n._1.value = n._2.asInstanceOf[INDArray]}
    graph.foreach{ _.forward() }
    graph.reverse.foreach{ _.backward() }
    t.addAndGet(1)
    for( n <- trainables) {
      firstMoment(n).muli(beta1).addi(n.gradients(n).mul(1 - beta1))
      secondMoment(n).muli(beta2).addi( pow(n.gradients(n),2).mul(1 - beta2))
      val fhat = firstMoment(n).div(1 - math.pow(beta1, t.get))
      val shat = secondMoment(n).div(1 - math.pow(beta2, t.get))
      n.value.addi( fhat.mul(-stepSize).div(sqrt(shat).add(delta)))
    }
  }


  }
