import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark._

object svm {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      .set("spark.executor.memory", "3G")
      .set("spark.driver.cores", "4")

    val sc = new SparkContext(conf)

    
    val data = sc.textFile("/home/yxing/result_multi.svm").cache()
    val trainingData = data.map { line =>
      val parts = line.split(' ')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
    }

    
    // val testingData = MLUtils.loadLibSVMFile(sc, "/home/yxing/")
    
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(23)
      .run(trainingData)

    model.save(sc, "/home/yxing/model")
     
    println("Job Done!!!")

  }
}
