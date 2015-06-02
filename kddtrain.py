from __future__ import print_function
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from numpy import array

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

conf = (SparkConf()
        .setAppName("svm")
        .setMaster("spark://cgal-arch:7077")
        .set("spark.executor.memory", "6G"))
sc = SparkContext(conf=conf)

data = sc.textFile("/home/yxing/result.libsvm")
dataT= sc.textFile("/home/yxing/test.libsvm")
#data = sc.textFile("/opt/spark/data/mllib/sample_svm_data.txt")

parsedData = data.map(parsePoint)
testingData= data.map(parsePoint)
# parsedData.saveAsTextFile("/home/yxing/temp")

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data

labelsAndPredsTesting = testingData.map(lambda p: (p.label, model.predict(p.features)))

trainErr = labelsAndPredsTesting.filter(lambda (v, p): v != p).count() / float(parsedData.count())

tp = 0.0
tn = 0.0
fp = 0.0
fn = 0.0

for labels in labelsAndPredsTesting.collect():
    if labels[0] >= 0.5 and labels[1] > 0.5:
        tp = tp+1
    elif labels[0] >= 0.5 and labels[1] < 0.5:
        tn = tn+1
    elif labels[0] <= 0.5 and labels[1] > 0.5:
        fp = fp+1
    elif labels[0] <= 0.5 and labels[1] < 0.5:
        fn = fn+1

        
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print( tp, tn, fp, fn, precision, recall)

print("Training Error = " + str(trainErr))
