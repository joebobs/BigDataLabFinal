from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import pyspark.sql.functions as f
from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
from tabulate import tabulate

spark = SparkSession.builder.appName("NYC_County_prediction").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BROKER_IP = "10.182.0.2:9092"
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", BROKER_IP).option("subscribe","quickstart-events").load()

split_cols = f.split(df.value,',')
df = df.withColumn('Summons Number',split_cols.getItem(1))
df = df.withColumn('Issuer Command',split_cols.getItem(2))
df = df.withColumn('Violation_County',split_cols.getItem(3))
df = df.withColumn('Days Parking In Effect',split_cols.getItem(4))
df = df.withColumn('From Hours In Effect',split_cols.getItem(5))
df = df.withColumn('Vehicle Year',split_cols.getItem(6))
model = PipelineModel.load('gs://bd_project_joe/finalproject/model/')

df = df.withColumn('true_label',df['Violation_County'])

predictions = model.transform(df)

output_df = predictions[['Summons Number','Violation_County_Prediction','true_label']]

def foreach_batch_function(df, epoch_id):
    if df.count() > 0:
        dftemp = df.toPandas()
        acc = accuracy_score(dftemp['true_label'], dftemp['Violation_County_Prediction'])
        f1 = f1_score(dftemp['true_label'], dftemp['Violation_County_Prediction'], average='weighted')
        output = pd.DataFrame([['Accuracy', acc],['F1-Score', f1]], columns = ['Metric', 'Value'])
        print('-------------------------------------------')
        print("Batch:", epoch_id)
        print('-------------------------------------------')
        print(tabulate(dftemp, headers='keys', tablefmt='psql', showindex=False))
        print('Metics for Batch',epoch_id,'is:')
        print(tabulate(output, headers = 'keys', tablefmt='psql', showindex=False))
    # Transform and write batchDF
    pass

#query1 = output_df.writeStream.queryName("output").outputMode('append').format('console').start()

query = output_df.writeStream.foreachBatch(foreach_batch_function).start()   
query.awaitTermination()