# Add Spark Python Files to Python Path
import sys
import os
SPARK_HOME = "/usr/local/spark-0.9.1" # Set this to wherever you have compiled Spark
os.environ["SPARK_HOME"] = SPARK_HOME # Add Spark path
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" # Set Local IP
sys.path.append( SPARK_HOME + "/python") # Add python files to Python Path

import numpy as np
from pyspark import SparkConf, SparkContext
import pyspark

def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
         .setMaster("local") # run on local
         .setAppName("Logistic Regression") # Name of App
         .set("spark.executor.memory", "1g")) # Set 1 gig of memory
    sc = pyspark.SparkContext(conf=conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    label = feats[len(feats) - 1]
    feats = feats[: len(feats) - 1]
    feats.insert(0,label)
    features = np.array([float(feature) for feature in feats]) # need floats
    return features


class myLogisticRegression(object):
    def __init__(self, data, lr=1.5, iter=200):
        self.w = np.zeros(shape=data.first().size-1)
        for _ in range(iter):
            grad = data.map(lambda x: self.gradMapper(x)).reduce(lambda a, b: a + b)
            self.w -= lr * grad / data.count()
    
    def __sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def gradMapper(self, p):
        x = p[1:]
        y = p[0]
        grad = -self.__sigmoid(-y * np.dot(self.w, x)) * y * x
        return grad
    
    def predict(self, x):
        yh = np.asscalar(np.dot(self.w, x))
        return self.__sigmoid(yh)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("gs://r09922114-bucket/data_banknote_authentication.txt")
parsedData = data.map(mapper)

# Train model
model = myLogisticRegression(parsedData, 2.0, 500)

# Predict the first elem will be actual data and the second 
# item will be the prediction of the model
labelsAndPreds = parsedData.map(lambda point: (int(point[0]), 
        round(model.predict(point[1:]))))

# Evaluating the model on training data
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())

# Print some stuff
print("Training Error = " + str(trainErr))

print(parsedData)
print(model)