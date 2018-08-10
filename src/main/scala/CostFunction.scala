package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp}


import com.github.timsetsfire.nn.node._

package object costfunctions {

    /** Mean square error Cost Node
      *
      * @param y actual values
      * @param yhat predicted values
      */
    class MSE(y: Node, yhat: Node) extends Node(List(y,yhat)) {

      var diff = null.asInstanceOf[INDArray]

      override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = this.inboundNodes(1).value
        val obs = y.shape.apply(0).toDouble
        this.diff = y - yhat
        this. value = (this.diff * this.diff).sum(0).sum(1) / (obs.toDouble)
      }

      override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val obs = this.inboundNodes(0).value.shape.apply(0).toDouble
        this.gradients(this.inboundNodes(0)) = this.diff * (2/obs)
        this.gradients(this.inboundNodes(1)) = this.diff * (-2/obs)
      }

    }

    /** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */
    object MSE {
      def apply(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new MSE(y, yhat)
    }

    class BCE(y: Node, yhat: Node) extends Node(List(y,yhat)) {

      var diff = null.asInstanceOf[INDArray]

      override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = this.inboundNodes(1).value
        val obs = y.shape.apply(0).toDouble
        this.diff = (y / yhat) + ( y.mul(-1) + 1d) / (yhat.mul(-1) + 1d)
        val temp = ((y * log(yhat))) + ((y.mul(-1) + 1d)*log(yhat.mul(-1)+1d))
  	    this.value = temp.sum(0).div(obs.toDouble).mul(-1)
      }

      override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = this.inboundNodes(1).value
        val obs = y.shape.apply(0).toDouble
        this.gradients(this.inboundNodes(0)) = (log(yhat) - log( yhat.sub(1).mul(-1))).div(-obs)
        //this.gradients(this.inboundNodes(1)) = (y - yhat).div(-obs)
        this.gradients(this.inboundNodes(1)) = ((y - yhat) / (yhat - yhat*yhat)).div(-obs)

      }
    }


    class BceWithLogits(y: Node, logits: Node) extends Node(List(y,logits)) {

      var diff = null.asInstanceOf[INDArray]

      override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = sigmoid(this.inboundNodes(1).value)
        val obs = y.shape.apply(0).toDouble
        val temp = ((y * log(yhat))) + ((y.mul(-1) + 1d)*log(yhat.mul(-1)+1d))
        this.value = temp.sum(0).div(obs.toDouble).mul(-1)
      }

      override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = sigmoid(this.inboundNodes(1).value)
        val obs = y.shape.apply(0).toDouble
        this.gradients(this.inboundNodes(0)) = (log(yhat) - log( yhat.sub(1).mul(-1))).div(-obs)
        this.gradients(this.inboundNodes(1)) = (y - yhat).div(-obs)

      }
    }


  // https://deepnotes.io/softmax-crossentropy
  // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    class CrossEntropyWithLogits(y: Node, logits: Node) extends Node(List(y,logits)) {
      var diff = null.asInstanceOf[INDArray]

      override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val logits = this.inboundNodes(1).value
        val p = exp(logits)
        p.diviColumnVector( p.sum(1))
        val obs = y.shape.apply(0).toDouble
        this.value = (y * log(p)).sum(0).sum(1).div(-obs)
      }
      override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val logits = this.inboundNodes(1).value
        val p = exp(logits)
        p.diviColumnVector( p.sum(1))
        val obs = y.shape.apply(0).toDouble
        this.gradients(this.inboundNodes(0)) = log(p).div(-obs)
        this.gradients(this.inboundNodes(1)) = (y - p).div(-obs)
      }
    }


    class GenBCE(yhat: Node) extends Node(List(yhat)) {

      var diff = null.asInstanceOf[INDArray]

      override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = this.inboundNodes(1).value
        val obs = y.shape.apply(0).toDouble
        // this.diff = (y / yhat) + ( y.mul(-1) + 1d) / (yhat.mul(-1) + 1d)
        val temp = log(yhat.mul(-1).add(1d))
  	    this.value = temp.sum(0).div(obs.toDouble).mul(-1)
      }

      override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        val y = this.inboundNodes(0).value
        val yhat = this.inboundNodes(1).value
        val obs = y.shape.apply(0).toDouble
        this.gradients(this.inboundNodes(0)) = Nd4j.zerosLike(yhat)
        //this.gradients(this.inboundNodes(1)) = (y - yhat).div(-obs)
        this.gradients(this.inboundNodes(1)) = (Nd4j.onesLike(yhat) / yhat).div(-obs)

      }
    }
}
