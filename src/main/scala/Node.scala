package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map=>MutMap}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms.{sigmoid, tanh, relu, log, exp}



/** a node in a directed graph
  *
  * @constructor Create a new node with a list of `inboundNodes`
  * @param inboundNodes The inbound nodes for this node
  * @author Timothy Whittaker
  * @version 1.0
  * @see [[https://medium.com/udacity/the-miniflow-lesson-929200f72e27]] and [[https://gist.github.com/jychstar/7aa4751c369fb296b53e33ec788e88bd]]
  * for more information
  */
  package object node {


  class Node(val inboundNodes: List[Node] = List()) {

    var name: String = null

    def setName(s: String): Unit = {
      this.name = s
    }

    override def toString = {
      if(name == null) s"${this.getClass.getSimpleName}@${java.lang.Integer.toHexString(this.hashCode())}"
      else s"{${name}}@${java.lang.Integer.toHexString(this.hashCode())}"
    }

    /** size
      * this correspond to the size of the node's value.
      * the value is an INDArray from Nd4j.
      * the value is an size._1 by size._2 INDArray
      * will need to be cast as an Int when used
      */

      val size: (Any, Any) = (None, None)

      /** value
        * placeholder for node's value
        */
      var value = null.asInstanceOf[INDArray]

      /** outboundNodes
        * outbound nodes for this node
        */
        val outboundNodes = new ArrayBuffer[Node]

      /** gradients for the outbound nodes and current nodes
        * used for Backpropogation
        */
      val gradients: MutMap[Node, INDArray] = MutMap()

      // update outbound nodes of the nbound node to include this node
      inboundNodes.foreach { n => n.outboundNodes += this }

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
        * @return Returns MatMul Node.  This is meant to mimic matrix multiplication
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

    class Transpose(x: Node) extends Node(List(x)) {

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
    class Add(x: Node, y: Node) extends Node(List(x,y)) {

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

    class MatMul(x: Node, y: Node) extends Node(List(x,y)) {

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

    class Input(override val size: (Any, Any) = (None, None)) extends Node()

    class Placeholder(override val size: (Any, Any) = (None, None)) extends Input(size)

    class Variable(override val size: (Any, Any) = (None, None), val initialize: String = "xavier") extends Input(size)


}
