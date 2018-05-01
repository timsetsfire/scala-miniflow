package com.github.timsetsfire.nn

import scala.collection.mutable.{Map => MutMap, Set => MutSet, ArrayBuffer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import com.github.timsetsfire.nn.node._

  // L ← Empty list that will contain the sorted elements
  // S ← Set of all nodes with no incoming edge
  // while S is non-empty do
  //     remove a node n from S
  //     add n to tail of L
  //     for each node m with an edge e from n to m do
  //         remove edge e from the graph
  //         if m has no other incoming edges then
  //             insert m into S
  // if graph has edges then
  //     return error (graph has at least one cycle)
  // else
  //     return L (a topologically sorted order)
  // reference https://gist.github.com/ThiporKong/4399695
  // reference https://en.wikipedia.org/wiki/Topological_sorting

package object graph {

  class DirectedAcyclicGraph(val x: MutMap[Node, ArrayBuffer[Node]])

  def topologicalSort(inputs: MutMap[Node, ArrayBuffer[Node]],
    sorted: ArrayBuffer[Node]=ArrayBuffer()
  ): ArrayBuffer[Node] = {
    val (inputNodesMap, otherNodesMap) = inputs.partition{ _._2.isEmpty}
    if(inputNodesMap.isEmpty) {
      if(otherNodesMap.isEmpty) sorted else sys.error("There is likely a cycle")
    } else {
      val inputNodes = inputNodesMap.map{ _._1 }
      val next = MutMap() ++ otherNodesMap.mapValues{ inputs => inputs -- inputNodes }
      topologicalSort( next, sorted ++ inputNodes)
    }
  }

}
