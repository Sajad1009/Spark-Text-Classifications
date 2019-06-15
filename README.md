# Spark-Text-Classifications

## Conferences Spark Text Classifications according to the titles of submitted papers


Most of the times its time consuming for the acadmic researchers to choose a suitable academic conference to submit his/her  academic papers. We define “suitable conference”, meaning the conference is aligned with the researcher’s work and have a good academic ranking.

Using the conference proceeding data set, we are going to categorize research papers by conferences. Let’s get started. The data set can be found here.


You sould have spark installed in your server or your computer for the example here the spark installed in google could servers.
For more information you can check the following link

https://cloud.google.com/dataproc/docs/tutorials/jupyter-notebook#create_a_bucket_in_your_project
https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52


### Import the required libaries to your spark  
----- Here we using jupyter notebook ------------
```
import findspark # you need to install this one by pip install findspark 
findspark.init('/home/sak/spark-2.4.3-bin-hadoop2.7') # set your path 
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('lr_example').getOrCreate()
from pyspark import SparkContext 
sc= spark.sparkContext
import pandas as pd
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import string
import re
from collections import Counter
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

```
### Importing the data from your computer/server if using hadoop server you need to load the files to the HDFS server the global server 
```
data = spark.read.csv("research_paper.csv", inferSchema=True, header=True) 
data.head(5)
-----------------------------------------------------------------------------------------------------------------------------------------+--------------------+----------+
|               Title|Conference|
+--------------------+----------+
|Innovation in Dat...|      VLDB|
|High performance ...|     ISCAS|
|enchanted scissor...|  SIGGRAPH|
|Detection of chan...|   INFOCOM|
|Pinning a Complex...|     ISCAS|
|Analysis and Desi...|     ISCAS|
|Dynamic bluescreens.|  SIGGRAPH|
|A Quantitative As...|   INFOCOM|
|Automatic sanitiz...|       WWW|
|A &#916;&#931; IR...|     ISCAS|
|Architecture of a...|     ISCAS|
|Rule-based Servic...|       WWW|
|Business Policy M...|      VLDB|
|A high speed and ...|     ISCAS|
|PREDIcT: Towards ...|      VLDB|
|SocialSensor: sen...|       WWW|
|Parametric keyfra...|  SIGGRAPH|
|An Explanation fo...|   INFOCOM|
|Hot Block Cluster...|      VLDB|
|Analysis of propa...|     ISCAS|
+--------------------+----------+
only showing top 20 rows


```
```
data.printSchema()
----------------------------------------------------------------------------------------------------------------------------------------
root
 |-- Title: string (nullable = true)
 |-- Conference: string (nullable = true)

```
Count the number of Conferances 

```
from pyspark.sql.functions import col 
data.groupBy("Conference") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()
-------------------------------------------------------------------------------------------------------------------------------------    
+----------+-----+
|Conference|count|
+----------+-----+
|     ISCAS|  864|
|   INFOCOM|  515|
|      VLDB|  423|
|       WWW|  379|
|  SIGGRAPH|  326|
+----------+-----+    
    
```

### Check if  there is null value 
```
from pyspark.sql.functions import isnan, when, count, col

data.select([count(when(isnan(c), c)).alias(c) for c in data.columns]).show()
----------------------------------------------------------------------------------------------------------------------------------------
+-----+----------+
|Title|Conference|
+-----+----------+
|    0|         0|
+-----+----------+
```
Clean the Texts of the columns using re

```
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

def clean_text(c):
  c = lower(c)
  c = regexp_replace(c, "^rt ", "")
  c = regexp_replace(c, "(https?\://)\S+", "")
  c = regexp_replace(c, "[^a-zA-Z0-9\\s]", "")
  

  return c

data1 = data.select((clean_text(col("title")).alias("title")), ((clean_text(col("Conference")).alias("Conference"))))

data1.show(5)
--------------------------------------------------------------------------------------------------------------------------------------
+--------------------+----------+
|               title|Conference|
+--------------------+----------+
|innovation in dat...|      vldb|
|high performance ...|     iscas|
|enchanted scissor...|  siggraph|
|detection of chan...|   infocom|
|pinning a complex...|     iscas|
+--------------------+----------+
only showing top 5 rows
```
```
## processing the data

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="Title", outputCol="words", pattern="\\W")
# stop words
remover = StopWordsRemover()
stopwords = remover.getStopWords()  
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(stopwords)
# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)

label_stringIdx = StringIndexer(inputCol = "Conference", outputCol = "label")
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors, label_stringIdx])
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

--------------------------------------------------------------------------------------------------------------------------------------
+--------------------+----------+--------------------+--------------------+--------------------+-----+
|               Title|Conference|               words|            filtered|            features|label|
+--------------------+----------+--------------------+--------------------+--------------------+-----+
|Innovation in Dat...|      VLDB|[innovation, in, ...|[innovation, data...|(791,[32,40,184,4...|  2.0|
|High performance ...|     ISCAS|[high, performanc...|[high, performanc...|(791,[16,42,301,3...|  0.0|
|enchanted scissor...|  SIGGRAPH|[enchanted, sciss...|[enchanted, sciss...|(791,[87,330,405]...|  4.0|
|Detection of chan...|   INFOCOM|[detection, of, c...|[detection, chann...|(791,[1,38,46,80,...|  1.0|
|Pinning a Complex...|     ISCAS|[pinning, a, comp...|[pinning, complex...|(791,[13,103,782]...|  0.0|
+--------------------+----------+--------------------+--------------------+--------------------+-----+
only showing top 5 rows

```
``` 
# split the data and count them 
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed = 100)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
---------------------------------------------------------------------------------------------------------------------------------------
Training Dataset Count: 1756
Test Dataset Count: 751
``` 
### Using Machine learning (Logstic Regression) 
``` 
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("Title","Conference","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
------------------------------------------------------------------------------------------------------------------------------------


+------------------------------+----------+------------------------------+-----+----------+
|                         Title|Conference|                   probability|label|prediction|
+------------------------------+----------+------------------------------+-----+----------+
|Front-end amplifier of low-...|     ISCAS|[0.9873925466446979,0.00332...|  0.0|       0.0|
|Low-voltage SOI CMOS DTMOS/...|     ISCAS|[0.9742686991022053,0.00850...|  0.0|       0.0|
|A low power multi-mode CMOS...|     ISCAS|[0.9674492923760447,0.01337...|  0.0|       0.0|
|P<sup>2</sup>E-DWT: A paral...|     ISCAS|[0.9659249230169219,0.00634...|  0.0|       0.0|
|A continuous-time band-pass...|     ISCAS|[0.962968104908483,0.008284...|  0.0|       0.0|
|WL-VC SRAM: a low leakage m...|     ISCAS|[0.9606127886621211,0.00716...|  0.0|       0.0|
|Design of process variation...|     ISCAS|[0.9604440270461447,0.01587...|  0.0|       0.0|
|Split-ADC Digital Backgroun...|     ISCAS|[0.9579038172854173,0.00757...|  0.0|       0.0|
|Code division parallel delt...|     ISCAS|[0.9568771961418872,0.00538...|  0.0|       0.0|
|Improved hybrid coding sche...|     ISCAS|[0.956053637454894,0.013767...|  0.0|       0.0|
+------------------------------+----------+------------------------------+-----+----------+
only showing top 10 rows

``` 
### Apply MulticlassClassificationEvaluator

```
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
----------------------------------------------------------------------------------------------------------------------------------
0.7650884770005698

```
### You can add more machine learning classifiers such ad DTree and Rforest, naive base ...etc to get better results 




