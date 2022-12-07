# Databricks notebook source
# MAGIC %md This notebook series is also available at https://github.com/databricks-industry-solutions/ab-testing.

# COMMAND ----------

# MAGIC %pip install mlflow>=1.29.0 numpy>=1.23.4

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait until there is data available to make predictions

# COMMAND ----------

import time
# Check that the streaming table exists
while True:
  if spark._jsparkSession.catalog().tableExists("solacc_ab_test", "risk_stream_source"):
    break
  else:
    time.sleep(1)

minimum_number_records = 1
while True:
  if spark.read.table("solacc_ab_test.risk_stream_source").count() >= minimum_number_records:
    break
  else:
    time.sleep(1)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Real time risk prediciton with A/B testing
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC In this notebook we will use the following:
# MAGIC - The two models trained previously (models A and B) that are registered on MLflow
# MAGIC - The streaming Delta table created before
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_3.png?raw=true" width="1000"/>
# MAGIC 
# MAGIC 
# MAGIC When the streaming data comes in, 50% of the data will go on the fly to model A and 50% to model B. The predictions would be served in a streaming fashion to a client, perhaps with a Kafka server, although that is not included in this demo, but we should see how easy it would be. What we do instead is to save these predictions in a Delta table. This table will later on be used to compute the quality metrics using the ground truth, and we will display it in Databricks SQL.
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/abtesting.png?raw=true"/>

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the models
# MAGIC Once we have registered our models we can see them in the MLflow UI. Note the version numbers of your models. In this case we will use versions 1 and 2, although for you these might be different. You can change this in the next cell.
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/model_versions.png?raw=true" width="1000"/>

# COMMAND ----------

import pyspark.sql.functions as F
import mlflow

model_a_version = 1
model_b_version = 2
model_name = "german_credit"

model_a = mlflow.spark.load_model(
  model_uri=f"models:/{model_name}/{model_a_version}" # Logistic regression model
)

model_b = mlflow.spark.load_model(
  model_uri=f"models:/{model_name}/{model_b_version}" # Gradient boosting
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Read the streaming data

# COMMAND ----------

df = (
  spark
  .readStream
  .format("delta")
  .table("solacc_ab_test.risk_stream_source")
  .withColumn("timestamp", F.unix_timestamp(F.current_timestamp()))
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC If we wanted to read a stream from Kafka instead, the code would be very similar:
# MAGIC ~~~
# MAGIC df = (
# MAGIC   spark 
# MAGIC   .readStream 
# MAGIC   .format("kafka") 
# MAGIC   .option("kafka.bootstrap.servers", "host1:port1,host2:port2") 
# MAGIC   .option("subscribe", "topic1") 
# MAGIC   .load()
# MAGIC )
# MAGIC ~~~
# MAGIC 
# MAGIC More info at https://docs.databricks.com/structured-streaming/kafka.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Divide the incoming credit requests into two equally sized groups: A and B
# MAGIC We will use a random number generator to make the split. 50% of the request will be made with model B and the remaining 50% with model A.

# COMMAND ----------

b_model_fraction = 0.5

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf

@pandas_udf('double')
def pandas_random_number(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: np.random.uniform())

df_with_split = df.withColumn("random_number", pandas_random_number("id"))
  
df_a = (
  df_with_split
  .where(F.col("random_number") >= b_model_fraction)
  .withColumn("group", F.lit("A"))
)

df_b = (
  df_with_split
  .where(F.col("random_number") < b_model_fraction)
  .withColumn("group", F.lit("B"))
)

display(df_a.union(df_b))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions with the two models
# MAGIC - Model A will make predictions on the data classified in group A
# MAGIC - Model B will to the same for the data in group B

# COMMAND ----------

cols_keep = ["group", "id", "prediction", "probability", "timestamp"]

df_pred_a = (
  model_a
  .transform(df_a)
  .select(cols_keep)
)

df_pred_b = (
  model_b
  .transform(df_b)
  .select(cols_keep)
)

df_pred = df_pred_a.union(df_pred_b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to a Delta Table while streaming

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS solacc_ab_test.risk_stream_predictions;

# COMMAND ----------

dbutils.fs.rm("/FileStore/tmp/streaming_ckpnt_risk_demo", recurse=True)

# COMMAND ----------

(
  df_pred
  .writeStream
  .format("delta")
  .option("checkpointLocation", f"/FileStore/tmp/streaming_ckpnt_risk_demo")
  .table("solacc_ab_test.risk_stream_predictions")
)

# COMMAND ----------

# MAGIC %md
# MAGIC If we wanted to write to a stream from Kafka instead, the code would be very similar:
# MAGIC ~~~
# MAGIC (
# MAGIC   df 
# MAGIC   .writeStream 
# MAGIC   .format("kafka") 
# MAGIC   .option("kafka.bootstrap.servers", "host1:port1,host2:port2") 
# MAGIC   .option("topic", "topic1") 
# MAGIC   .start()
# MAGIC )
# MAGIC ~~~

# COMMAND ----------

display(
  spark
  .readStream
  .table("solacc_ab_test.risk_stream_predictions")
)

# COMMAND ----------

# MAGIC %md Now let's gracefully terminate the streaming queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gracefully stop the streams when most data points have predictions

# COMMAND ----------

minimum_number_records = 300 # users can set this number up to 400 based on our toy test data from notebook3
while True:
  current_number_records = spark.read.table("solacc_ab_test.risk_stream_predictions").count()
  print("Number of records with predictions", current_number_records)
  if current_number_records >= minimum_number_records:
    break
  else:
    time.sleep(10)

for s in spark.streams.active:
  print("Stopping stream")
  s.stop()

