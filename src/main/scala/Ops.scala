package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp, abs, sqrt, pow}


import com.github.timsetsfire.nn.node._

package object ops {

  class MeanOp(x: Node) extends Node(x) {

      override def forward(value: INDArray = null): Unit = {
        val in = inboundNodes(0)
        this.value = in.value.mean(0)
      }
      override def backward(value: INDArray = null): Unit = {
        this.inboundNodes.foreach{
          n =>
            val Array(rows, cols) = this.x.value.shape
            this.gradients(n) = Nd4j.zeros(rows, cols)
        }
          this.outboundNodes.foreach{
            n =>
            val gradCost = n.gradients(this)
            this.gradients(this.inboundNodes(0)) +=  Nd4j.onesLike(this.inboundNodes(0).value).div(this.inboundNodes(0).value.shape.apply(0))
          }
      }
  }

  class VarianceOp(x: Node, mean: MeanOp) extends Node(x, mean) {

      override def forward(value: INDArray = null): Unit = {
        val in = inboundNodes(0)
        this.value = pow(in.value.std(0),2)
      }
      override def backward(value: INDArray = null): Unit = {
        this.inboundNodes.foreach{
          n =>
            val Array(rows, cols) = this.x.value.shape
            this.gradients(n) = Nd4j.zeros(rows, cols)
        }
          this.outboundNodes.foreach{
            n =>
            val gradCost = n.gradients(this)
            this.gradients(this.inboundNodes(0)) +=  (x.value subRowVector mean.value).div(x.value.shape.apply(0)).mul(2)
            this.gradients(this.inboundNodes(1)) +=  (x.value subRowVector mean.value).div(x.value.shape.apply(0)).mul(-2)
          }
      }
  }

  class Standardize(x: Node, mean: MeanOp, variance: VarianceOp, val epsilon: Double = 1e-8) extends Node(x, mean, variance) {

    override def forward(value: INDArray = null): Unit = {
      val in = inboundNodes(0)
      this.value = (x.value subRowVector mean.value).divRowVector(sqrt(variance.value.add(epsilon)))
    }
    override def backward(value: INDArray = null): Unit = {
      this.inboundNodes.foreach{
        n =>
          val Array(rows, cols) = this.value.shape
          this.gradients(n) = Nd4j.zeros(rows, cols)
      }
        this.outboundNodes.foreach{
          n =>
          val gradCost = n.gradients(this)
          this.gradients(this.inboundNodes(0)) +=  (Nd4j.onesLike(x.value).divRowVector( sqrt(variance.value.add(epsilon))))
          this.gradients(this.inboundNodes(1)) +=  (Nd4j.onesLike(x.value).mul(-1).divRowVector( sqrt(variance.value.add(epsilon))))
          this.gradients(this.inboundNodes(2)) +=  (x.value.subRowVector(mean.value)).mul(-1).divRowVector( pow( sqrt(variance.value.add(epsilon)), 3))
        }
    }
  }

  object Standardize {

    def apply(x: Node, epsilon: Double=1e-8) = {
      val m = new MeanOp(x)
      val s = new VarianceOp(x,m)
      new Standardize(x, m, s, epsilon)
    }
  }

}
