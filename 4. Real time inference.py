# Databricks notebook source
# MAGIC %md ---
# MAGIC title: A/B testing with MLflow 4 - Real time inference
# MAGIC authors:
# MAGIC -  Sergio Ballesteros
# MAGIC tags:
# MAGIC - machine-learning
# MAGIC - python
# MAGIC - pyspark
# MAGIC - a/b testing
# MAGIC - ab testing
# MAGIC - binary-classifier
# MAGIC - mllib
# MAGIC - credit risk
# MAGIC - loan risk
# MAGIC - finance
# MAGIC created_at: 2021-07-27
# MAGIC updated_at: 2021-07-27
# MAGIC tldr: Loads two trained ML models and performs online inference on a stream of data coming from a Delta table. Predictions will be used for the A/B test.
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook Links
# MAGIC - AWS demo.cloud: [https://demo.cloud.databricks.com/#notebook/10781570](https://demo.cloud.databricks.com/#notebook/10781570)

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
  .table("risk_stream_source")
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Divide the incoming credit requests into two equally sized groups: A and B
# MAGIC We will use the ID of the request to make the split. Other logics would also work, for example with a random number generator we could choose a different split

# COMMAND ----------

df_with_split = df.withColumn("even", F.col("id")%2)
df_a = (
  df_with_split
  .where(F.col("even") == 0)
  .withColumn("group", F.lit("A"))
)

df_b = (
  df_with_split
  .where(F.col("even") != 0)
  .withColumn("group", F.lit("B"))
)

display(df_a.union(df_b))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make predictions with the two models
# MAGIC - Model A will make predictions on the data classified in group A
# MAGIC - Model B will to the same for the data in group B

# COMMAND ----------

df_pred_a = (
  model_a
  .transform(df_a)
  .select("group", "id", "prediction", "probability")
)

df_pred_b = (
  model_b
  .transform(df_b)
  .select("group", "id", "prediction", "probability")
)

df_pred = df_pred_a.union(df_pred_b)
display(df_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to a Delta Table while streaming

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS risk_stream_predictions;

# COMMAND ----------

dbutils.fs.rm("/FileStore/tmp/streaming_ckpnt_risk_demo", recurse=True)
(
  df_pred
  .withColumn("timestamp", F.unix_timestamp(F.current_timestamp()))
  .writeStream
  .format("delta")
  .option("checkpointLocation", f"/FileStore/tmp/streaming_ckpnt_risk_demo")
  .table("risk_stream_predictions")
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

display(spark.readStream.table("risk_stream_predictions"))
