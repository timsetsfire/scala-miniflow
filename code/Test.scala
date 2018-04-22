import org.nd4s.Implicits._
import org.nd4j.linalg.factory.Nd4j
// import org.datavec.api.

// datavec examples
// val x = Nd4j.readNumpy("c:/users/whittakert/desktop/nd4s/resources/smallMatrix.txt", " ")


val x = Nd4j.readNumpy("c:/users/whittakert/desktop/nd4s/resources/xMatrix.csv", ",")
val y = Nd4j.readNumpy("c:/users/whittakert/desktop/nd4s/resources/yMatrix.csv", ",")

//
// val x = Nd4j.randn(500,13)
// val y = Nd4j.randn(500,1)
val w = Nd4j.zeros(x.columns, y.columns)
val b = Array(0d).toNDArray
val Array(m,n) = x.shape

// stdize
val xs = x.subRowVector(x.mean(0)).divRowVector( x.std(0))
val ys = y.subRowVector(y.mean(0)).divRowVector( y.std(0))


import scala.math.{abs, exp}
def sigmoid(z: Double) = 1d / (1 + exp(-z))
def signum(x: Double) = if(x==0) 0 else x / abs(x)
def softThresh(b: Double, gamma: Double) = {
  val sig = signum(b);
  val f = abs(b) - gamma  ;
  val pos = ( abs(f) + f ) /2;
  sig * pos;
}

def cost(x: INDArray, y: INDArray)(b: INDArray) = {
  val Array(m,n) = x.shape
  val error = (y - x.mmul(b))
  error.norm2Number.doubleValue / m
}

// coordinate descent
val i = 0
val alpha = 1d
val lambda = 0.00
for(j <- 0 until 100) {
  for(i <- 0 until n) {
    val error = ys - xs.mmul(w)
    val update = xs(->, i).transpose.mmul( error + xs(->,i).mul(w(i,0))) / m
    w(i, 0) = softThresh(update(0,0) ,  alpha * lambda) / ( 1 + (1 - alpha)*lambda)
  }
}


val b = parms.b(ind)
val wx = (x(::, ind) *:* w) *:* s
val t = 1/ (x.rows.toDouble) *( wx.t * ( (y - g(x,parms) + (parms.b(ind)*x(::, ind)) )))
parms.b(ind) = softThresh(t, alpha * lambda) / ( 1/x.rows.toDouble * (wx.t*x(::, ind)) + (1 - alpha)*lambda )
if( abs( b - parms.b(ind)) < tolerance) Unit
else descend(parms, ind)


for(i <- 0 to 1000) {
  val gradBias = Nd4j.ones(1,m).mmul(ys - xs.mmul(w).add(b(0,0))).div(m).mul(0.01)
  val gradWeights = xs.transpose.mmul( (ys - xs.mmul(w).add(b(0,0)) )).div(m).mul(0.01)
  w.addi(gradWeights)
  b.addi(gradBias)
}


val arr = (1 to 9).asNDArray(3,3)
val sub = arr(0 -> 2, 1 -> 3)


// does not support int - do make explicit what you are doing
val x = Nd4j.create( Array[Float](1,2,3,4), Array(2,2))
// or
val x = Nd4j.create( Array(1f,2f,3f,4f), Array(2,2))

// zeros
val z = Nd4j.zeros(10,2)
// add 10 to each element
// holy shit - the following is inplace
val tens = z.addi(10)

// uniform random numbers
Nd4j.rand(10, 4) // 2 dimensinos 10 x 4
Nd4j.rand( Array(10, 4, 4)) // 10 x 4 x 4


val t1 = System.currentTimeMillis
for(i <- 0 until 100) {
  val x = Nd4j.rand( Array(10000, 100)) // 10 x 4 x 4
  x.transpose.mmul(x)
}
val t2 = System.currentTimeMillis
println( s"${(t2 - t1)/1e3} seconds")

// test file reading
val readFromText = Nd4j.readNumpy(makeResourcePath("/resources/matrix.txt"));
//

// set seed
Nd4j.getRandom.setSeed(10L)

// creating nd arrays from scala arrays
import org.nd4j.linalg.util.ArrayUtil
val myDoubleArray = Array(Array(1,2,3), Array(10,11,12))
ArrayUtil.flatten(myDoubleArray)

// copy the arrays
val y = x.dup
x.addi(10)
println(x)
println(y)

// creating arrays
val nRows = 2;
val nColumns = 2;
// Create INDArray of zeros
val zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
val ones = Nd4j.ones(nRows, nColumns);
//hstack
val hstack = Nd4j.hstack(ones,zeros);
println("### HSTACK ####");
println(hstack);

// concat
val nRows = 2;
val nColumns = 2;
//INDArray of zeros
val zeros = Nd4j.zeros(nRows, nColumns);
// Create one of all ones
val ones = Nd4j.ones(nRows, nColumns);
// Concat on dimension 0
val combined = Nd4j.concat(0,zeros,ones);
println("### COMBINED dimension 0####");
println(combined);
//Concat on dimension 1
val combined2 = Nd4j.concat(1,zeros,ones);
println("### COMBINED dimension 1 ####");
println(combined2);

// linear space
Nd4j.linspace(10, 20, 20 - 10 + 1)
Nd4j.linspace(1,25,25).reshape(5,5)

// not very idiomatic scalav

import org.nd4j.linalg.api.iter.NdIndexIterator
val myArray = Nd4j.create( Array(1d,2d,3d,4d))
val (nRows, nCols) = (myArray.rows, myArray.columns)
val iter = new NdIndexIterator(nRows, nCols);
while (iter.hasNext()) {
    val nextIndex = iter.next();
    val nextVal = myArray.getDouble(nextIndex:_*);
    println(nextVal)
}

import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4j.linalg.ops._

import org.nd4j.linalg.ops.impl.transforms._
import org.nd4j.linalg.api.ops.impl.transforms._

new Tanh(x)

Nd4j.getExecutioner().execAndReturn(new Tanh(x))
Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh",x))


val x = Nd4j.randn(30,3)
val y = Nd4j.randn(30,1)
val b = Nd4j.ones(30,1)
val X = N4dj.hstack(b,x)

InvertMatrix.invert( X.transpose mmul X) mmul X.transpose mmul y
