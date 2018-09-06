package com.github.timsetsfire.nn.batchnormalization

import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp}
import com.github.timsetsfire.nn.node._
import com.github.timsetsfire.nn.ops._

import java.util.Calendar

class BatchNormalization(activation: Node, gamma: Variable, beta: Variable) extends Node(activation, gamma, beta) {

    override def forward(value: INDArray = null): Unit = {
      this.value = (activation.value.mulRowVector( gamma.value ).addRowVector(beta.value))
    }

    override def backward(value: INDArray = null): Unit = {

      this.inboundNodes.foreach{
        n =>
          this.gradients(n) = Nd4j.zerosLike(n.value)
      }
      if(value == null) {
        this.gradients(this) = value
        this.outboundNodes.foreach{
          n =>
          val gradCost = n.gradients(this)
          this.gradients(this.inboundNodes(0)) +=  (gradCost mmul Nd4j.diag(gamma.value))
          this.gradients(this.inboundNodes(1)) +=  ((activation.value mmul Nd4j.onesLike(gamma.value.transpose)).transpose mmul gradCost)
          this.gradients(this.inboundNodes(2)) +=  (Nd4j.ones(activation.value.shape.apply(0),1).transpose mmul gradCost)
        }
      } else {
        this.gradients(this) = value
        val gradCost = this.gradients(this)
        this.gradients(this.inboundNodes(0)) +=  (gradCost mmul Nd4j.diag(gamma.value))
        this.gradients(this.inboundNodes(1)) +=  ((activation.value mmul Nd4j.onesLike(gamma.value.transpose)).transpose mmul gradCost)
        this.gradients(this.inboundNodes(2)) +=  (Nd4j.ones(activation.value.shape.apply(0),1).transpose mmul gradCost)
      }
    }

}

object BatchNormalization {
  def apply(activation: Node, size: (Any, Any) = (None, None), epsilon: Double=1e-8) = {
    val s = Standardize(activation, epsilon)
    val gamma = new Variable(size = (1, size._2.asInstanceOf[Int]))
    val beta = new Variable(size = (1, size._2.asInstanceOf[Int]))
    new BatchNormalization(s, gamma, beta)
  }
}
