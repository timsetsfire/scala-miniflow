
package com.github.timsetsfire.nn.optimize

import scala.collection.mutable.ArrayBuffer
import com.github.timsetsfire.nn.node._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._



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
