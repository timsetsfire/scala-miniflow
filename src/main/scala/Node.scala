


import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu}
import scala.reflect.BeanProperty

/**
  * A class to represent the general idea of a node in a directed acyclic graph
  * Each node has a value, a set of inbound nodes, and a set of outbound nodes
  *
  * Specify the `inboundNodes` when creating a new `Node`
  * @constructor Create a new node with a list of `inboundNodes`
  * @param inboundNodes Then inbound nodes for this node
  * @author Timothy Whittaker
  * @version 1.0
  * @see [[https://medium.com/udacity/the-miniflow-lesson-929200f72e27]] and [[https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd]]
  * for more information
  */
class Node(val inboundNodes: List[Node] = List()) {

  var value = null.asInstanceOf[INDArray]
  val outboundNodes = new ArrayBuffer[Node]
  val gradients: MutMap[Node, INDArray] = MutMap()
  // update outbound nodes of inputs to include this
  /**
    * @return Returns Unit.  Is only meant to update outbound nodes of
    * this node's inbound nodes to include this node
    */
  def setOutboundNodes(): Unit = {
    for(n <- inboundNodes) n.outboundNodes += this
  }

  /**
    * @return Returns Unit.  This method does change this node's  state by
    * updated this node's value.
    * This is called during forward propogation
    */
  def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    if(value != null) {
      this.value = value
    }
  }
  /** Backpropogation step for this node
    * @return Returns Unit.  This method does change this node's state by
    * updating this nodes' value.
    * This method is called during backward propogation.
    * Each node type has its own forward method.
    */
  def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
      this.gradients(this) = Nd4j.zeros(this.value.shape:_*)
      this.outboundNodes.foreach{
        n =>
          val gradCost = n.gradients(this)
          this.gradients(this) += gradCost * 1d
      }
    }

  /**
    * @return Retusn MatMul Node.  This is meant to mimic matrix multiplication
    * @param n right multiply this node by n
    * @todo Set up the forward and backward methods in MatMul
    */
  def *(n: Node) = {
    /**
    * mutiply
    * meant to act as a matrix multiply between
    * two nodes with values of appropriate shape
    * @param n is node which will right multiply this
    */
    new MatMul(this, n)
  }

  /**
    * @return Returns Add Node.  This is meant to mimic matrix addition
    * @todo Set up the forward and backward methods in Add
    * @param n is a node which will be added to this
    */
  def +(n: Node) = {
    new Add(this, n)
  }

  /**
    * @return Returns Transpose Node.  This is meant to mimic matrix transposition
    * @todo Set up the forward and backward methods in Tranpose
    */
  def T() = {
    new Transpose(this)
  }
}


/** Tranpose Node
  * @constructor Create a new `Transpose` node by specifying the node to Transpose.
  * This was not meant to be constructed directly.
  * @param x The node to transpose.
  * @example val x = new Node()
  */
class Transpose(x: Node) extends Node(List(x)) {
  setOutboundNodes
  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val x = inboundNodes(0).value
    this.value = x.transpose
  }
}


/** Tranpose Node
  * @constructor Create a new `Transpose` node by specifying the node to Transpose.
  * This was not meant to be constructed directly.
  * @param x The node to transpose.
  * @example val x = new Node()
  */
class Add(x: Node, y: Node) extends Node(List(x,y)) {
  setOutboundNodes
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
  * This was not meant to be constructed directly.
  * @param x left node in matrix mulitply
  * @param y right node in matrix multiply
  */
class MatMul(x: Node, y: Node) extends Node(List(x,y)) {
  setOutboundNodes
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

/** General Input node
  * These are really comparable to tensorflow.PlaceHolder
  */
class Input() extends Node
object Input {
  def apply() = new Input()
}


// define some activation functions
class Linear(inputs: Node, weights: Node, bias: Node) extends Node(List(inputs, weights, bias)) {
  setOutboundNodes
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
object Linear {
  def apply(inputs: Node, weights: Node, bias: Node) = new Linear(inputs, weights, bias)
}

class Sigmoid(node: Node) extends Node(List(node)) {
  setOutboundNodes
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
object Sigmoid {
  def apply(node: Node) = new Sigmoid(node)
}

class Tanh(node: Node) extends Node(List(node)) {
  setOutboundNodes
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
        this.gradients(this.inboundNodes(0)) += (out * out).mul(-1d).add(1d)
    }
  }
}
object Tanh {
  def apply(node: Node) = new Tanh(node)
}

class Relu(node: Node) extends Node(List(node)) {
  setOutboundNodes
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
        this.gradients(this.inboundNodes(0)) += out.gt(Nd4j.zerosLike(out))
    }
  }
}
object Relu {
  def apply(node: Node) = new Relu(node)
}


class MSE(y: Node, yhat: Node) extends Node(List(y,yhat)) {
  setOutboundNodes
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
    this.gradients(this.inboundNodes(0)) = this.diff * (2 / obs)
    this.gradients(this.inboundNodes(1)) = this.diff * (-2/obs)
  }
}
object MSE {
  def apply(y: Node, yhat: Node) = new MSE(y, yhat)
}
