package com.github.timsetsfire.nn

import scala.collection.mutable.{ArrayBuffer, Map => MutMap}
import com.github.timsetsfire.nn.node.Node

package object session {

  class Session {
    val graph = MutMap[Node, ArrayBuffer[Node]]()
  }


  def createSession(graph: MutMap[Node, ArrayBuffer[Node]])(actions: MutMap[Node, ArrayBuffer[Node]] => Unit) = {
    actions(graph)
    graph
  }

}
