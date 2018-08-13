//
//
package com.github.timsetsfire.nn.regularization

import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp}
import com.github.timsetsfire.nn.node.Node

import org.nd4j.linalg.api.rng.distribution.impl.BinomialDistribution
import java.util.Calendar



class Dropout[T <: Node]( node: T, val dropout: Double, val seed: Long = Calendar.getInstance.getTime.getTime , var train: Boolean = true) extends Node(List(node)) {

  Nd4j.getRandom().setSeed(seed)
  val b = new BinomialDistribution(1, 1d-dropout)
  var sample: INDArray = null

  override def forward(value: INDArray = null): Unit = {
    val sampleSize = inboundNodes(0).value.shape
    this.sample = if(train) b.sample(sampleSize) else Nd4j.ones(sampleSize:_*)
    this.value = (this.sample * node.value).div(1-dropout)
  }

  override def backward(value: INDArray = null): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = inboundNodes(0).value.shape
        this.gradients(n) = Nd4j.zeros(rows, cols)
    }
    if(value == null) {
      this.outboundNodes.foreach{
        n =>
        val gradCost = n.gradients(this)
        this.gradients(this.inboundNodes(0)) +=  this.sample.div(1-dropout) * gradCost
      }
    } else {
      this.gradients(this) = value
      val gradCost = this.gradients(this)
      this.gradients(this.inboundNodes(0)) +=  this.sample.div(1-dropout) * gradCost
    }
  }
}
