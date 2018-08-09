/** Create SoftMax activation node
  *
  * @param node inbound node
  */
class SoftMax(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) extends Node(List(node))(graph) {

  override def forward(value: INDArray = null.asInstanceOf[INDArray]): Unit = {
    val in = inboundNodes(0)
    this.value = exp(value)
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
  def apply(node: Node)(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = new SoftMax(node)(graph)
  def apply(node: Node, size: (Any, Any), initialize: String = "xavier")(implicit graph: MutMap[Node, ArrayBuffer[Node]]) = {
    val l1 = Linear(node, size, initialize)(graph)
    new SoftMax(l1)(graph)
  }
}
