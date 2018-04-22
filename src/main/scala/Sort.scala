import scala.collection.mutable.{Map => MutMap, Set => MutSet, ArrayBuffer}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray


class DAG(x: List[Node])

object TopologicalSort {


  def topologicalSort(feedDict: Map[Node, Any]) = {

    val l = ArrayBuffer[Node]() // empty list that will contain the sorted elements
    val s: MutSet[Node] = MutSet(feedDict.keys.toList:_*) // set of nodes with no incoming edges

    val nodes = s.clone // making a copy
    val graph = MutMap[Node, MutMap[String, MutSet[Node]]]() // empty graph

    while(!nodes.isEmpty) {
      val n = nodes.head
      nodes -= n
      if(!graph.isDefinedAt(n)) {
        graph.update(n, MutMap( "in" -> MutSet(), "out" -> MutSet()))
      }
      n.outboundNodes.foreach{ m =>
        if(!graph.isDefinedAt(m)) {
          graph.update(m, MutMap( "in" -> MutSet(), "out" -> MutSet()))
        }
        graph(m)("in") += n
        graph(n)("out") += m
        nodes += m
      }
    }

    while(s.size > 0) {
      val n = s.head
      s -= n

      if(n.isInstanceOf[Input]) n.value = feedDict(n).asInstanceOf[INDArray]

      l += n
      n.outboundNodes.foreach{ m =>
        graph(n)("out").remove(m)
        graph(m)("in").remove(n)
        if(graph(m)("in").isEmpty) s += m
      }

    }

    l
  }


}
