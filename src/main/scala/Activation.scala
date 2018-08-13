package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp, abs}


import com.github.timsetsfire.nn.node._

package object activation {

// Linear Node
class Linear(inputs: Node,
             weights: Input,
             bias: Input) extends Node(List(inputs, weights, bias)) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val List(x, w, b) = inboundNodes.map{ _.value}
    this.value = (x mmul w) addRowVector b
  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = n.value.shape
        this.gradients(n) = Nd4j.zeros(rows, cols)
    }
    this.outboundNodes.foreach{
      n =>
        val gradCost = n.gradients(this)
        this.gradients(this.inboundNodes(0)) += (gradCost mmul this.inboundNodes(1).value.transpose)
        this.gradients(this.inboundNodes(1)) += (this.inboundNodes(0).value.transpose mmul gradCost)
        this.gradients(this.inboundNodes(2)) += gradCost.sum(0)
    }
  }
}

// need a maxout activation node

/** Factory for [[com.github.timsetsfire.nn.node.Linear]] instances. */
object Linear {
  /** create a Linear node with a given input node, weight node and bias node
    *
    * given inputs X, weights w and biases b, this node
    * yields value = (X*w)+b
    * @param input input nodes
    * @param w weights
    * @param b bias
    * @param name name
    */
  def apply[T <: Node](input: T, w: Variable, b: Variable) = new Linear(input, w, b)

  /** create a linear node wiht a given input and setSize
    *
    * @param input input node
    * @param size a tupe of Any.  size._1 is the size of the input and size._2 is size of output
    */
  def apply[T <: Node](input: T, size: (Any, Any)) = {
      val (in, out) = size match {
        case (x: Int, y: Int) => (x, y)
        case (None, y: Int) => {
          if(input.size._2 == 0) throw new Exception(s"input has size ${input.size}, and size provided is ${size} are not valid")
          else (input.size._2, y)
        }
        case _ => throw new Exception(s"input has size ${input.size}, and size provided is ${size} are not valid")
      }
      val w = new Variable( (in, out))
      val b = new Variable( (1, out))
      new Linear(input, w, b)
  }
}


/** Create Sigmoid activation node
  *
  * @param node inbound node
  */
class Sigmoid(node: Node) extends Node(List(node)) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val in = inboundNodes(0)
    this.value = sigmoid(in.value)
  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = this.value.shape
        this.gradients(n) = Nd4j.zeros(rows, cols)
    }
    if(value == null) {
      this.outboundNodes.foreach{
        n =>
        val gradCost = n.gradients(this)
        val sigmoid = this.value
        this.gradients(this.inboundNodes(0)) +=  sigmoid * (sigmoid.mul(-1d) + 1d) * gradCost
      }
    } else {
      this.gradients(this) = value
      val gradCost = this.gradients(this)
      val sigmoid = this.value
      this.gradients(this.inboundNodes(0)) +=  sigmoid * (sigmoid.mul(-1d) + 1d) * gradCost
    }
  }
}

/** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */
object Sigmoid {
  def apply(node: Node) = new Sigmoid(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new Sigmoid(l1)
  }
}

/** Create SoftMax activation node
  *
  * @param node inbound node
  */
class SoftMax(node: Node) extends Node(List(node)) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val in = inboundNodes(0)
    this.value = exp(in.value)
    this.value.diviColumnVector( this.value.sum(1))

  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = this.value.shape
        this.gradients(n) = Nd4j.zeros(rows, cols)
    }
    this.outboundNodes.foreach{
      n =>
        val gradCost = n.gradients(this)
        val softmax = this.value
        this.gradients(this.inboundNodes(0)) += softmax * (softmax.mul(-1d) + 1d) * gradCost
    }
  }
}

/** Factory for [[com.github.timsetsfire.nn.node.SoftMax]] instances. */
object SoftMax {
  def apply(node: Node) = new SoftMax(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new SoftMax(l1)
  }
}


/** Create Tanh activation node
  *
  * @param node inbound node
  */
class Tanh(node: Node) extends Node(List(node)) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val in = inboundNodes(0)
    this.value = tanh(in.value)
  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = this.value.shape
        this.gradients(n) = Nd4j.zeros(rows,cols)
    }
    this.outboundNodes.foreach{
      n =>
        val gradCost = n.gradients(this)
        val out = this.value
        this.gradients(this.inboundNodes(0)) += (out * out).mul(-1d).add(1d)*gradCost
    }
  }
}

/** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */

object Tanh {
  def apply(node: Node) = new Tanh(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new Tanh(l1)
  }
}

/** Create Relu activation node
  *
  * @param node inbound node
  */
class ReLU(node: Node) extends Node(List(node)) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val in = inboundNodes(0)
    this.value = relu(in.value)
  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.inboundNodes.foreach{
      n =>
        val Array(rows, cols) = this.value.shape
        this.gradients(n) = Nd4j.zeros(rows,cols)
    }
    if(value == null) {
      this.outboundNodes.foreach{
        n =>
        val gradCost = n.gradients(this)
        val out = this.value
        this.gradients(this.inboundNodes(0)) += out.gt(Nd4j.zerosLike(out))*gradCost
      }
    } else {
      this.gradients(this) = value
      val gradCost = this.gradients(this)
      val out = this.value
      this.gradients(this.inboundNodes(0)) += out.gt(Nd4j.zerosLike(out))*gradCost
    }
  }
}

/** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */

object ReLU {
  def apply(node: Node) = new ReLU(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new ReLU(l1)
  }
}

class Maxout(node: Node) extends Node(List(node)) {

  override def forward(value: INDArray = null): Unit = {
    val in = inboundNodes(0)
    val m = in.value.max(1)
    this.value = (in.value.addColumnVector(m).div(2)) add (abs( in.value.subColumnVector(m))).div(2)
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
        val out = this.value
        val in = this.inboundNodes(0).value
        this.gradients(this.inboundNodes(0)) += (out eq in)*gradCost
    }
  }
}

object Maxout {
  def apply(node: Node) = new Maxout(node)
  def apply(node: Node, size: (Any, Any)) = {
    val l1 = Linear(node, size)
    new Maxout(l1)
  }
}
}
