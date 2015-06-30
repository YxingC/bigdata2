import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark._
import scala.io.Source
import scala.util.control.Breaks._
import math._

object testing {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("svm Classification")
      .set("spark.executor.memory", "6G")
      .set("spark.driver.cores", "4")

    val sc = new SparkContext(conf)
    val modelPath = "/home/yxing/Model_d"
    val testFile = "/home/yxing/input/kddcup.test.fivelabel.svm"
    //val testFile = "/home/yxing/input/kddcup.test.label.dos.svm"
   // val numOfData = sc.textFile(testFile).count()
/*

    val model_n = SVMModel.load(sc, "/home/yxing/Model_n_nonSVD")
    val model_d = SVMModel.load(sc, "/home/yxing/Model_d_nonSVD")
    val model_u = SVMModel.load(sc, "/home/yxing/Model_u_nonSVD")
    val model_r = SVMModel.load(sc, "/home/yxing/Model_r_nonSVD")
    val model_p = SVMModel.load(sc, "/home/yxing/Model_p_nonSVD")
    val model_L = LogisticRegressionModel.load(sc, "/home/yxing/model_L_nonSVD")
    //val th = model.getThreshold
    model_n.clearThreshold()
    model_d.clearThreshold()
    model_u.clearThreshold()
    model_r.clearThreshold()
    model_p.clearThreshold()

    val data = sc.textFile("/home/yxing/input/kddcup.test.label.r2l.svm")
    val testingData = data.map { line =>
      val parts =  line.split(" ")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(_.toDouble)))
    }
    val scoreAndLabels = testingData.map { point =>
      val score = model_r.predict(point.features)
      (score, point.label)
    }
    val fileOutput = scoreAndLabels
      .map(row => Array(row._1, row._2))
      .map(row => row.mkString(" "))

    fileOutput.saveAsTextFile("/home/yxing/output")

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    //println("Threshold = " + th)
    println("Area under ROC = " + auROC)
    */
 
/*
    val tempArray = Array.ofDim[Double](5)
    val confusionMx = Array.ofDim[Int](5, 5)

    for(line <- Source.fromFile("/home/yxing/input/kddcup.test.fivelabel.svm").getLines())
    {
      val parts = line.split(" ")
      val labeledPt = LabeledPoint(parts(0).toDouble, Vectors.dense(parts.tail.map(_.toDouble)))
      tempArray(0) = model_n.predict(labeledPt.features)
      tempArray(1) = model_d.predict(labeledPt.features)
      tempArray(2) = model_u.predict(labeledPt.features)
      tempArray(3) = model_r.predict(labeledPt.features)
      tempArray(4) = model_p.predict(labeledPt.features)
      var maxIdx = -1.0
      var maxTmp = -1.0
      for(i <- 0 to 4)
      {
        breakable
        {
          if (tempArray(i) < 0)
            break  // break out of the 'breakable', continue the outside loop
          else if(maxTmp < tempArray(i))
          {
            maxIdx = i
            maxTmp = tempArray(i)
          }
        }
      }

     
      if(maxIdx.toInt == -1)
      {
        maxIdx = model_L.predict(labeledPt.features)
      }

      val classIdx = maxIdx.toInt
      val trueIdx = labeledPt.label.toInt

      if(classIdx == trueIdx)
      {
        confusionMx(classIdx)(classIdx) = confusionMx(classIdx)(classIdx)+1
      }
      else
      {
        if(trueIdx > -1 && trueIdx < 5)
          confusionMx(trueIdx)(classIdx) = confusionMx(trueIdx)(classIdx)+1
      }

    }

    for(i <- 0 to 4)
    {
      for(j <- 0 to 4)
      {
        print(confusionMx(i)(j) + " ")
      }
      println
    }

 */
/*     

    val scoreAndLabels = testingData.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val fileOutput = scoreAndLabels
      .map(row => Array(row._1, row._2))
      .map(row => row.mkString(" "))

    fileOutput.saveAsTextFile("/home/yxing/output")

    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    //println("Threshold = " + th)
    println("Area under ROC = " + auROC)

 */

    val model = LogisticRegressionModel.load(sc, "/home/yxing/Model_L")
    val data = sc.textFile("/home/yxing/input/kddcup.test.fivelabel.svm")
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

    val confusionMx = metrics.confusionMatrix

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
      println("[" + r + "] p: " + precision + " r:" + recall)
//      println("[" + r + "] " +  precision + " " + metrics.precision(r.toDouble))
    }

    for(r <- 0 to 4)
    {
      for(c <- 0 to 4)
      {
        if(c == 4)
          print(confusionMx.apply(r, c))
        else
          print(confusionMx.apply(r, c) + " ")
      }
      println
    }






    //println("Precision = " + precision)
    //println("Recall = " + recall)

    println("Done!!!")


  }
}
