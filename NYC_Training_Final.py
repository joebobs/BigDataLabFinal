
# coding: utf-8

# In[1]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, IndexToString, FeatureHasher
import pyspark.sql.functions as f
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
start_time = time.time()


print('Libraries imported')

# In[ ]:


spark = SparkSession.builder         .appName("Text Categorization")        .getOrCreate()
df = (spark.read
         .format("com.databricks.spark.csv")
         .option("header", "true")
         .load("gs://bdl2021_final_project/nyc_tickets_train.csv").limit(20000))

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Data Read:', time_elapsed)


# In[12]:


'''df = spark.read.option("header","true")      .csv("gs://bd_project_joe/nyc_tickets_train.csv").limit(100)'''


# In[28]:


df2 = df.drop(*['Time First Observed', 'Intersecting Street', 'Law Section', 'Violation Legal Code', 
                'From Hours In Effect', 'To Hours In Effect', 'Unregistered Vehicle?', 
                'Meter Number', 'Violation Description', 'No Standing or Stopping Violation', 'Hydrant Violation', 
                'Double Parking Violation', 'Latitude', 'Longitude', 'Community Board', 
                'Community Council', 'Census Tract', 'BIN', 'BBL', 'NTA',  # REst (under) are not removed due to NULL
                'Summons Number', 'Plate ID', 'Vehicle Expiration Date',     
                'Issue Date', 'Street Code1', 'Street Code2', 'Street Code3', 'Date First Observed', # rest (under) removed for testing
                'Issuer Squad', 
                'Violation Time', 'Violation In Front Of Or Opposite', 
                'House Number', 'Street Name', 
                'Sub Division', 'Days Parking In Effect', 'Vehicle Color', 'Vehicle Year',
                'Feet From Curb'])

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Rows Dropped:', time_elapsed)

# In[ ]:


(trainingData_original, testData) = df2.randomSplit([0.7, 0.3], seed = 100)
trainingData = trainingData_original

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Split performed:', time_elapsed)


# Categorical Encoding

# In[29]:


cat_cols = ['Registration State', 'Plate Type', 'Violation Code', 'Vehicle Body Type', 'Vehicle Make', 
            'Issuing Agency', 'Issuer Code', 'Issuer Command']
''', 'Issuer Squad', 
'Violation Time', 'Violation In Front Of Or Opposite', 
'House Number', 'Street Name', 
'Sub Division', 'Days Parking In Effect', 'Vehicle Color', 'Vehicle Year',
'Feet From Curb']
'''


# In[30]:


cat_cols_indexed = map(lambda x: x+'_Index', cat_cols)
cat_cols_onehot = map(lambda x: x+'_Onehot', cat_cols)


# In[ ]:


#featureIndexers = FeatureHasher(inputCols=cat_cols,outputCol='vector')
#trainingData = featureIndexers.transform(trainingData)

featureIndexers = []
for i in cat_cols:
    featureIndexers.append(StringIndexer(inputCol=i,outputCol=i+'_Index').setHandleInvalid("keep").fit(trainingData))

count = 0
for i in cat_cols:
    trainingData = featureIndexers[count].transform(trainingData)
    count += 1

end_time = time.time()
time_elapsed = (end_time - start_time)
print('String Indexer done:', time_elapsed)
# In[40]:

OHE = OneHotEncoderEstimator(inputCols=cat_cols_indexed,outputCols=cat_cols_onehot).fit(trainingData)
trainingData = OHE.transform(trainingData)

# In[41]:


columns = ['Summons Number', 'Plate ID', 'Registration State', 'Plate Type', 
           'Issue Date', 'Violation Code', 'Vehicle Body Type', 'Vehicle Make', 
           'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 
           'Vehicle Expiration Date', 'Issuer Code', 'Issuer Command', 
           'Issuer Squad', 'Violation Time', 'Violation_County', 'Violation In Front Of Or Opposite', 
           'House Number', 'Street Name', 'Date First Observed', 'Sub Division', 
           'Days Parking In Effect', 'Vehicle Color', 'Vehicle Year', 'Feet From Curb', 'Violation Post Code']

'''for i in cat_cols:
    columns.remove(i)
columns = columns + cat_cols_onehot
columns.remove('Violation_County') # feature to predict


for i in ['Summons Number', 'Plate ID', 'Vehicle Expiration Date', 
          'Issue Date', 'Street Code1', 'Street Code2', 'Street Code3',
          'Date First Observed']:
    columns.remove(i)
columns'''
columns = cat_cols_onehot


# In[42]:


assembler = VectorAssembler(inputCols=columns,
                           outputCol='vector')
trainingData = assembler.transform(trainingData)


# In[46]:


labelIndexer = StringIndexer().setInputCol('Violation_County').setOutputCol("label").setHandleInvalid("skip").fit(trainingData)
trainingData = labelIndexer.transform(trainingData)

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Output Labels Created:', time_elapsed)
# In[47]:


classifier = RandomForestClassifier(featuresCol='vector',labelCol='label').fit(trainingData)
trainingData = classifier.transform(trainingData)

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Model Fit:', time_elapsed)
# In[48]:


outputLabel = IndexToString().setInputCol("prediction").setOutputCol('Violation_County_Prediction').setLabels(labelIndexer.labels)

end_time = time.time()
time_elapsed = (end_time - start_time)
print("All pre pipeline tasks are done:", time_elapsed)

# In[49]:


pipeline = Pipeline(stages= featureIndexers + [OHE, assembler, labelIndexer, classifier, outputLabel]) #OHE, assembler, 


# In[ ]:

trainingData_original = trainingData_original.limit(1)


model = pipeline.fit(trainingData_original)

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Pipeline has been fit:', time_elapsed)

# In[ ]:


predictions =  model.transform(testData)


# In[ ]:


evaluator =  MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))
print("Test Error = %g " % (1.0 - accuracy))


# In[ ]:


model.save('gs://joebobby/finalproject/model')

end_time = time.time()
time_elapsed = (end_time - start_time)
print('Model Saved:', time_elapsed)

