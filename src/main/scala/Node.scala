


import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu}



class Node(val inboundNodes: List[Node] = List()) {

  var value = null.asInstanceOf[INDArray]
  val outboundNodes = new ArrayBuffer[Node]
  val gradients: MutMap[Node, INDArray] = MutMap()
  // update outbound nodes of inputs to include this
  def setOutboundNodes(): Unit = {
    for(n <- inboundNodes) n.outboundNodes += this
  }
  def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    if(value != null) {
      this.value = value
    }
  }
  def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = ???

  def *(n: Node) = {
    /**
    * mutiply
    * meant to act as a matrix multiply between
    * two nodes with values of appropriate shape
    * @param n is node which will right multiply this
    */
    new MatMul(this, n)
  }

  def +(n: Node) = {
    /**
    * add
    * meant to act as addition of two inboundNodes
    * with appropriately shaped value
    * @param n is a node which will be added to this
    **/
    new Add(this, n)
  }

  def T() = {
    /**
    * transpose
    * meant to act as transposition
    **/
    new Transpose(this)
  }
}

class Transpose(x: Node) extends Node(List(x)) {
  setOutboundNodes
  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val x = inboundNodes(0).value
    this.value = x.transpose
  }
}
class Add(x: Node, y: Node) extends Node(List(x,y)) {
  setOutboundNodes
  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val x = inboundNodes(0).value
    val y = inboundNodes(1).value
    this.value = x + y
  }
}
class MatMul(x: Node, y: Node) extends Node(List(x,y)) {
  setOutboundNodes
  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val x = inboundNodes(0).value
    val y = inboundNodes(1).value
    this.value = x mmul y
  }
}

class Input() extends Node {
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    this.gradients(this) = Nd4j.zeros(1,1)
    this.outboundNodes.foreach{
      n =>
        val gradCost = n.gradients(this)
        this.gradients(this) += gradCost * 1d
    }
  }
}
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
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = ???
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
    this. value = this.diff.norm2(0) / 10d
  }
  override def backward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val obs = this.inboundNodes(0).value.shape.apply(0).toDouble
    this.gradients(this.inboundNodes(0)) = this.diff * (2 / obs)
    this.gradients(this.inboundNodes(1)) = this.diff * (-2/obs)
  }
}
