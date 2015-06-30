import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix,Matrices}
import org.apache.spark.mllib.linalg.{Vector,Vectors}
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark._

object svd {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      .set("spark.executor.memory", "6G")
      .set("spark.driver.cores", "4")

    val sc = new SparkContext(conf)
    val featureFile = "/home/yxing/input/kddcup.data_unlabeled.txt"

    //val A: RowMatrix = svd(featureFile, sc)

    //train(A, "/home/yxing/input/kddcup.data.normal.label", "/home/yxing/Model_n", sc)
    //train(A, "/home/yxing/input/kddcup.data.dos.label", "/home/yxing/Model_d", sc)
    //train(A, "/home/yxing/input/kddcup.data.u2r.label", "/home/yxing/Model_u", sc)
    //train(A, "/home/yxing/input/kddcup.data.r2l.label", "/home/yxing/Model_r", sc)
    //train(A, "/home/yxing/input/kddcup.data.probe.label", "/home/yxing/Model_p", sc)
    //trainL(A, "/home/yxing/input/kddcup.data.five.label", "/home/yxing/Model_L", sc)
    trainWithoutSVD(featureFile, "/home/yxing/input/kddcup.data.probe.label", "/home/yxing/Model_p_nonSVD", sc)

    println("Done")
    
  }

  def svd(featureFile:String, sc:SparkContext) : RowMatrix = {
    val data_f = sc.textFile(featureFile)
    val unlabeledData = data_f.map(l => Vectors.dense(l.split(' ').map(_.toDouble)))
    val mat: RowMatrix = new RowMatrix(unlabeledData)
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(20, computeU=true)
    val A : RowMatrix = svd.U.multiply(Matrices.diag(svd.s)).multiply(svd.V.transpose)

    return A
  }

  def train(A:RowMatrix, labelFile:String, outputPath:String, sc:SparkContext) {

    val labelData = sc.textFile(labelFile)

    val features = A.rows.map(v => v.toArray).zipWithIndex.map(p => (p._2, p._1))
    val labels = labelData.map(l => l.toDouble)
      .map(x => Array(x))
      .zipWithIndex
      .map(p => (p._2, p._1))

    val trainingData = (labels union features)
      .reduceByKey(_ ++ _)
      .sortByKey()
      .map(x => x._2)
      .map(A => LabeledPoint(A.head, Vectors.dense(A.tail)))

    val numIterations = 100
    val model = SVMWithSGD.train(trainingData, numIterations)

    model.save(sc, outputPath)
  }

  def trainL(A:RowMatrix, labelFile:String, outputPath:String, sc:SparkContext) {

    val labelData = sc.textFile(labelFile)

    val features = A.rows.map(v => v.toArray).zipWithIndex.map(p => (p._2, p._1))
    val labels = labelData.map(l => l.toDouble)
      .map(x => Array(x))
      .zipWithIndex
      .map(p => (p._2, p._1))

    val trainingData = (labels union features)
      .reduceByKey(_ ++ _)
      .sortByKey()
      .map(x => x._2)
      .map(A => LabeledPoint(A.head, Vectors.dense(A.tail)))

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(5)
      .run(trainingData)

    model.save(sc, outputPath)
  }

  def trainWithoutSVD(featureFile:String, labelFile:String, outputPath:String, sc:SparkContext) {
    val labelData = sc.textFile(labelFile)
    val featureData = sc.textFile(featureFile)

    val features = featureData.map(line => line.split(" "))
      .map(A => A.map(_.toDouble))
      .zipWithIndex
      .map(p => (p._2, p._1))

    val labels = labelData.map(l => l.toDouble)
      .map(x => Array(x))
      .zipWithIndex
      .map(p => (p._2, p._1))

    val trainingData = (labels union features)
      .reduceByKey(_ ++ _)
      .sortByKey()
      .map(x => x._2)
      .map(A => LabeledPoint(A.head, Vectors.dense(A.tail)))

    val numIterations = 100
    val model = SVMWithSGD.train(trainingData, numIterations)

    model.save(sc, outputPath)
  }
}
