package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log}


package object node {

  /** A node in a directed graph
    *
    * @constructor Create a new node with a list of `inboundNodes`
    * @param inboundNodes The inbound nodes for this node
    * @param graph the graph param is an implicit and should be in the scope before
    * creating a Node, otherwise, it will create a new graph upon construction.
    * the default for graph is only temporary and will be removed in the future.
    * @author Timothy Whittaker
    * @version 1.0
    * @see [[https://medium.com/udacity/the-miniflow-lesson-929200f72e27]] and [[https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd]]
    * for more information
    */
  class Node(val inboundNodes: List[Node] = List())(implicit graph: MutMap[Node, ArrayBuffer[Node]] = MutMap()) {

    graph.update(this, ArrayBuffer())

    /** size
      * this corresponds to the size of the node's value
      * the value is an size._1 by size._2 Matrix
      */
    val size: (Any, Any) = (None, None)

    /** value
      * placeholder for the nodes value.
      */
    var value = null.asInstanceOf[INDArray]

    // outbound nodes for this node
    val outboundNodes = new ArrayBuffer[Node]

    // gradients for the outbound nodes and current nodes
    // used for Backpropogation
    val gradients: MutMap[Node, INDArray] = MutMap()

    /** update outbound nodes of the inbound node to include this node
      * update the graph as well
      */
    inboundNodes.foreach {
      n =>
        n.outboundNodes += this
        graph(this) += n
    }

    /** Forward propogration
      * @return Returns Unit.  This method changes this node's state by
      * @param value INDArray - this can probably be removed
      * This is called during forward propogation
      */
    def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      if(value != null) {
        this.value = value
      }
    }

    /** Backpropogation step for this node
      * @return Returns Unit.  This method does change this node's state by
      * @param value INDArray - this can probably be removed
      * This method is called during backward propogation.
      */
    def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
        this.gradients(this) = Nd4j.zeros(this.value.shape:_*)
        this.outboundNodes.foreach{
          n =>
            val gradCost = n.gradients(this)
            this.gradients(this) += gradCost * 1d
        }
      }

    /** multiply nodes
      * @return Retusn MatMul Node.  This is meant to mimic matrix multiplication
      * @param n right multiply this node by n
      */
    def *(n: Node) = {
      new MatMul(this, n)
    }

    /** add two nodes
      * @return Returns Add Node.  This is meant to mimic matrix addition
      * @param n is a node which will be added to this
      * for non-similarly shaped matrices, it obeys nd4j broadcasting.
      */
    def +(n: Node) = {
      new Add(this, n)
    }

    /** transpose a node
      * @return Returns Transpose Node.  This is meant to mimic matrix transposition
      * @todo Set up the forward and backward methods in Tranpose
      */
    def T() = {
      new Transpose(this)
    }
  }


  /** Tranpose Node
    * @constructor Create a new `Transpose` node by specifying the node to Transpose.
    * This is not meant to be used directly.
    * @param x The node to transpose.
    * @param graph
    */
  class Transpose(x: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(x))(graph) {

    override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      val x = inboundNodes(0).value
      this.value = x.transpose
    }
  }


  /** Add Node
    * @constructor Create a new `Add` node by specifying two nodes to add.
    * This is not meant to be used directly.
    * @param x
    * @param y
    */
  class Add(x: Node, y: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(x,y))(graph) {

    override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      val x = inboundNodes(0).value
      val y = inboundNodes(1).value
      this.value = x + y
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
          this.gradients(this.inboundNodes(0)) += gradCost
          this.gradients(this.inboundNodes(1)) += gradCost
      }
    }
  }

  /** Matrix Multiply Node
    * @constructor Create a new `MatMul` node by mulitplying the input nodes.  It
    * is expected that the input nodes are the appropriate shape.
    * This is not meant to be used directly.
    * @param x left node in matrix mulitply
    * @param y right node in matrix multiply
    */

  class MatMul(x: Node, y: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(x,y))(graph) {

    override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      val x = inboundNodes(0).value
      val y = inboundNodes(1).value
      this.value = x mmul y
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
      }
    }
  }

  /**
    *
    */
  class Input(
    override val size: (Any, Any) = (None, None)
  )(implicit graph: MutMap[Node, ArrayBuffer[Node]]=MutMap()) extends Node()(graph)

  class Placeholder(
    override val size: (Any, Any) = (None, None)
  )(implicit graph: MutMap[Node, ArrayBuffer[Node]]=MutMap()) extends Input(size)(graph)

  class Variable(
    override val size: (Any, Any) = (None, None),
    val initialize: String = "xavier"
  )(implicit graph: MutMap[Node, ArrayBuffer[Node]]=MutMap()) extends Input(size)(graph)

  // Linear Node
  class Linear(inputs: Node,
               weights: Input,
               bias: Input)(implicit graph: MutMap[Node, ArrayBuffer[Node]]=MutMap()) extends Node(List(inputs, weights, bias))(graph) {

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

  /** Factory for [[com.github.timsetsfire.nn.node.Linear]] instances. */
  object Linear {
    /** create a Linear node with a given input node, weight node and bias node
      *
      * given inputs X, weights w and biases b, this node
      * yields value = (X*w)+b
      * @param input input nodes
      * @param w weights
      * @param b bias
      */
    def apply[T <: Node](input: T, w: Variable, b: Variable)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new Linear(input, w, b)(graph)

    /** create a linear node wiht a given input and setSize
      *
      * @param input input node
      * @param size a tupe of Any.  size._1 is the size of the input and size._2 is size of output
      */
    def apply[T <: Node](input: T, size: (Any, Any), initialize: String = "xavier")(implicit graph: MutMap[Node, ArrayBuffer[Node]]=MutMap()) = {
        val (in, out) = size match {
          case (x: Int, y: Int) => (x, y)
          case (None, y: Int) => {
            if(input.size._2 == 0) throw new Exception(s"input has size ${input.size}, and size provided is ${size} are not valid")
            else (input.size._2, y)
          }
          case _ => throw new Exception(s"input has size ${input.size}, and size provided is ${size} are not valid")
        }
        val w = new Variable( (in, out), initialize)
        val b = new Variable( (1, out), initialize)
        new Linear(input, w, b)(graph)
    }
  }


  /** Create Sigmoid activation node
    *
    * @param node inbound node
    */
  class Sigmoid(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(node))(graph) {

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
      this.outboundNodes.foreach{
        n =>
          val gradCost = n.gradients(this)
          val sigmoid = this.value
          this.gradients(this.inboundNodes(0)) += sigmoid * (sigmoid.mul(-1d) + 1d) * gradCost
      }
    }
  }

  /** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */
  object Sigmoid {
    def apply(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new Sigmoid(node)(graph)
    def apply(node: Node, size: (Any, Any), initialize: String = "xavier")(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = {
      val l1 = Linear(node, size, initialize)(graph)
      new Sigmoid(l1)(graph)
    }
  }

  /** Create Tanh activation node
    *
    * @param node inbound node
    */
  class Tanh(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(node))(graph) {

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
    def apply(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new Tanh(node)(graph)
    def apply(node: Node, size: (Any, Any), initialize: String = "xavier")(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = {
      val l1 = Linear(node, size, initialize)(graph)
      new Tanh(l1)(graph)
    }
  }

  /** Create Relu activation node
    *
    * @param node inbound node
    */
  class ReLU(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(node))(graph) {

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
      this.outboundNodes.foreach{
        n =>
          val gradCost = n.gradients(this)
          val out = this.value
          this.gradients(this.inboundNodes(0)) += out.gt(Nd4j.zerosLike(out))*gradCost
      }
    }
  }

  /** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */

  object ReLU {
    def apply(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new ReLU(node)(graph)
    def apply(node: Node, size: (Any, Any), initialize: String = "xavier")(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = {
      val l1 = Linear(node, size, initialize)(graph)
      new ReLU(l1)(graph)
    }
  }


  /** Mean square error Cost Node
    *
    * @param y actual values
    * @param yhat predicted values
    */
  class MSE(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(y,yhat))(graph) {

    var diff = null.asInstanceOf[INDArray]

    override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      val y = this.inboundNodes(0).value
      val yhat = this.inboundNodes(1).value
      val obs = y.shape.apply(0).toDouble
      this.diff = y - yhat
      this. value = this.diff.norm2(0) / (obs.toDouble)
    }

    override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      val obs = this.inboundNodes(0).value.shape.apply(0).toDouble
      this.gradients(this.inboundNodes(0)) = this.diff * (2/obs)
      this.gradients(this.inboundNodes(1)) = this.diff * (-2/obs)
    }

  }

  /** Factory for [[com.github.timsetsfire.nn.node.Sigmoid]] instances. */
  object MSE {
    def apply(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new MSE(y, yhat)(graph)
  }

  class BCE(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(y,yhat))(graph) {

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

  class CrossEntropy(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(y,yhat))(graph) {



  }
  //     class BCE(y: Node, yhat: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(y,yhat))(graph) {
  //
  //   var diff = null.asInstanceOf[INDArray]
  //
  //   override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
  //     val y = this.inboundNodes(0).value
  //     val yhat = this.inboundNodes(1).value
  //     val obs = y.shape.apply(0).toDouble
  //     this.diff = (y / yhat) + ( y.mul(-1) + 1d) / (yhat.mul(-1) + 1d)
  //     val temp = ((y * log(yhat))) + ((y.mul(-1) + 1d)*log(yhat.mul(-1)+1d))
	//     this.value = temp.sum(0).div(obs.toDouble).mul(-1)
  //   }
  //
  //   override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
  //     val y = this.inboundNodes(0).value
  //     val yhat = this.inboundNodes(1).value
  //     val obs = y.shape.apply(0).toDouble
  //     this.gradients(this.inboundNodes(0)) = (log(yhat) - log( yhat.sub(1).mul(-1))).div(-obs)
  //     this.gradients(this.inboundNodes(1)) = ((y - yhat) / (yhat - yhat*yhat)).div(-obs)
  //   }
  //
  // }



  }
