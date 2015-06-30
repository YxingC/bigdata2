import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark._

object training {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      .set("spark.executor.memory", "3G")
      .set("spark.driver.cores", "4")

    val sc = new SparkContext(conf)

    
    val data4train = sc.textFile("/home/yxing/input/kddcup.data_unlabeled.txt")
    val labelData = sc.textFile("/home/yxing/input/kddcup.data.five.label")

    val features = data4train.map(line => line.split(" "))
      .map(A => A.map(_.toDouble))
      .zipWithIndex
      .map(p => (p._2, p._1))

    val labels = labelData.map(_.toDouble)
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

    model.save(sc, "/home/yxing/model_L_nonSVD")
     
    println("Job Done!!!")

  }
}
