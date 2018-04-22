
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.serde.binary.BinarySerde

import scala.util.{Try, Success, Failure}

import java.io._;
import java.nio.ByteBuffer;

val arrWrite = Nd4j.linspace(1,10,10);
val arrRead = null.asInstanceOf[INDArray];

//1. Binary format
//   Close the streams manually or use try with resources.
Try {
	(val sWrite = new DataOutputStream(new FileOutputStream(new File("resources/tmp.bin")))) {	Nd4j.write(arrWrite, sWrite) }
}

Try {
	(val sRead = new DataInputStream(new FileInputStream(new File("tmp.bin")))) {	arrRead = Nd4j.read(sRead)	}
}

//2. Binary format using java.nio.ByteBuffer;
// ByteBuffer buffer = BinarySerde.toByteBuffer(arrWrite);
// arrRead = BinarySerde.toArray(buffer);

//3. Text format
Nd4j.writeTxt(arrWrite, "resources/tmp.txt");
arrRead = Nd4j.readTxt("resources/tmp.txt");

// To read csv format:
// The writeNumpy method has been deprecated.
arrRead =Nd4j.readNumpy("tmp.csv", ", ");
