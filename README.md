# BigDataAnalytics Homework2

##Introduction

###Environment
####Spark 1.3.1
Method => Logistic Regression
Classes => 2(normal and abnormal)

####Execute
```
$SPARK_HOME/bin/spark-submit kddtrain.py
```
####Training Data (Kddcup99)
Training Data only use two labels(0 or 1)<br>
Testing Data use corrected data

## Result
**Precision = 0.998714041244**<br>
**Recall = 0.80158148911**<br>
**Training Error = 0.00431342199165**<br>
tp = 3909555, tn = 16095, fp = 5034, fn = 967747
3909555.0 16095.0 5034.0 967747.0  

