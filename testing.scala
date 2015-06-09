import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark._

object test {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      .set("spark.executor.memory", "3G")
      .set("spark.driver.cores", "4")

    val sc = new SparkContext(conf)

    val model = LogisticRegressionModel.load(sc, "/home/yxing/model")
    val data = sc.textFile("/home/yxing/test_multi.svm")
    val testingData = data.map { line =>
      val parts = line.split(' ')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
    }

    val predictLabels = testingData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictLabels)

    print(metrics)

    val precision = metrics.precision
    val recall = metrics.recall

    println("Precision = " + precision)
    println("Recall = " + recall)

    println("Job Done!!!")

  }
}
