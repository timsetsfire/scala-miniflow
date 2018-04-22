int numLinesToSkip = 0;
     char delimiter = ',';
     RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
     recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));

     //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
     int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
     int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
     int batchSize = 150;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

     DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
     DataSet allData = iterator.next();
     allData.shuffle();
     SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

     DataSet trainingData = testAndTrain.getTrain();
     DataSet testData = testAndTrain.getTest();

     //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
     DataNormalization normalizer = new NormalizerStandardize();
     normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
     normalizer.transform(trainingData);     //Apply normalization to the training data
     normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.datavec.api.records.reader.RecordReader


         val recordReader = new CSVRecordReader(0,",");
         recordReader.initialize(new FileSplit(new ClassPathResource("resources/IrisData/iris.txt").getFile()));
         //reader,label index,number of possible labels
         DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,4,3);
         //get the dataset using the record reader. The datasetiterator handles vectorization
         DataSet next = iterator.next();
         // Customizing params
         Nd4j.MAX_SLICES_TO_PRINT = 10;
         Nd4j.MAX_ELEMENTS_PER_SLICE = 10;
