import java.io._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark._

object testing {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      //.set("spark.executor.memory", "3G")
      //.set("spark.driver.cores", "4")
      .setMaster("spark://AhDa-PC:7077")//set spark master
      .setSparkHome( "C:\\Users\\AhDa\\Downloads\\spark-1.3.1-bin-hadoop2.4")//set spark home

    val sc = new SparkContext(conf)
    sc.addJar("lib/kddProject.jar")

    val model = LogisticRegressionModel.load(sc, "model")
    val data = sc.textFile("test_multi.svm")
    val testingData = data.map { line =>
      val parts = line.split(' ')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(x => x.toDouble)))
    }

    val predictLabels = testingData.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictLabels)

    //val precision = metrics.weightedPrecision
    //val recall = metrics.weightedRecall

    val confusionMx : Matrix  = metrics.confusionMatrix


    //println("rows = " + confusionMx.numRows)
    //println("cols = " + confusionMx.numCols)

    var r = 0
    var c = 0
    
    for(r <- 0 to confusionMx.numRows-1)
    {
      var precision = 0.0
      var recall = 0.0
      var tp = 0.0
      var fp = 0.0
      var fn = 0.0
      for(c <- 0 to confusionMx.numCols-1)
      {
        if(r == c)
        {
          tp = confusionMx.apply(r, c)
        }
        else
        {
          fp = fp + confusionMx.apply(c, r)
          fn = fn + confusionMx.apply(r, c)
        }
      }
      println
      if(tp == 0.0)
      {
        precision = 0.0
        recall = 0.0
      }
      else
      {
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
      }
      println("[" + r + "] " + precision)
//      println("[" + r + "] " +  precision + " " + metrics.precision(r.toDouble))
    }

    for(r <- 0 to 21)
    {
      for(c <- 0 to 21)
      {
        println(confusionMx.apply(c, r) + " ")
      }
      println
    }






    //println("Precision = " + precision)
    //println("Recall = " + recall)

    println("Job Done!!!")


  }
}
