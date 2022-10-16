# Databricks notebook source
# MAGIC %md This notebook series is also available at https://github.com/databricks-industry-solutions/ab-testing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Computing metrics
# MAGIC Great, now we have a table were we store the predictions and a table where we have the ground truth of the users who received predictions (we can assume that there is such a feedback loop).
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/arch_4.png?raw=true" width="1000"/>
# MAGIC 
# MAGIC 
# MAGIC In this notebook we are going to compare the predictions with the actual responses for the models A and B over time. We will compute the Precision Recall AUC in 1 minute buckets.
# MAGIC 
# MAGIC We will save these results in a Delta table so that we can read it from Databricks SQL. This will allow us to track the quality of both models over time and set up alerts when the quality of the models decrease over a certain threshold. This could be an input to retrain the models with fresher data. This process could be manual, but also could be easily automated by creating a job.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://miro.medium.com/max/3200/1*dCy-F02P3u4kWruKbz4FuA.png" width="1000"/>

# COMMAND ----------

import time
time.sleep(240) # this notebook needs to execute concurrently to notebook 3 and 4, but start a bit later than 4

# COMMAND ----------

# MAGIC %md
# MAGIC # Import libraries

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from scipy.stats import mannwhitneyu
from datetime import datetime
import pyspark.sql.types as T

# COMMAND ----------

# MAGIC %md
# MAGIC # Helper function

# COMMAND ----------

@pandas_udf("double", PandasUDFType.GROUPED_AGG)
def compute_metric(gt, p):
  precision, recall, thresholds = precision_recall_curve(gt, p)
  return auc(recall, precision)

# COMMAND ----------

# MAGIC %md
# MAGIC # Compute the Precision - Recall AUC metric

# COMMAND ----------

df_pred = (
  spark
  .read
  .table("solacc_ab_test.risk_stream_predictions")
  .select("group", "id", "prediction", vector_to_array(F.col("probability")).getItem(1).alias("prob"), "timestamp")
)

df_gt = (
  spark
  .read
  .table("solacc_ab_test.german_credit_data")
  .select("id", "risk")
  .withColumn("ground_truth", F.when(F.col("risk")=="good", 0).otherwise(1))
)

df_metrics = (
  df_gt
  .join(df_pred, on="id", how="inner")
  .withColumn("date_time", F.from_unixtime("timestamp", "MM-dd-yyyy HH:mm"))
  .groupby("group", "date_time")
  .agg(compute_metric("ground_truth", "prediction").alias("pr_auc"))
  .na
  .drop()
)
display(df_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Plot the above metrics

# COMMAND ----------

import plotly.express as px
pd1 = df_metrics.toPandas()
fig = px.line(pd1.sort_values(by=['date_time'], ascending=[True]), x='date_time', y='pr_auc', line_group='group', color='group')
fig

# COMMAND ----------

display(df_metrics.groupby("group").mean("pr_auc"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Statistical test
# MAGIC We can see that the model A has a lower PR AUC than the model B. We will use the Wilcoxon-Mann-Whitney test to check if this difference is statistically significant.
# MAGIC 
# MAGIC https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html

# COMMAND ----------


model_a_metric = df_metrics.where(F.col("group") == "A").select("pr_auc").toPandas()["pr_auc"].values
model_b_metric = df_metrics.where(F.col("group") == "B").select("pr_auc").toPandas()["pr_auc"].values

# Check if model B is better than model A
b_better_a = mannwhitneyu(
  x=model_a_metric,
  y=model_b_metric,
  alternative="less" # The distribution of the PR AUC of A is less than the PR AUC of B
)

# Check if model A is better than model B
a_better_b = mannwhitneyu(
  x=model_a_metric,
  y=model_b_metric,
  alternative="greater" # The distribution of the PR AUC of A is less than the PR AUC of B
)

print("Test model B better than model A", b_better_a)
print("Test model A better than model B", a_better_b)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save metrics and tests to a Delta Table and visualize them with Databricks SQL

# COMMAND ----------

if (a_better_b[1] < 0.05) and (b_better_a[1] > 0.05):
  best_model = "A"
  p_value = float(a_better_b[1])
elif (a_better_b[1] > 0.05) and (b_better_a[1] > 0.05):
  best_model = "A_or_B"
  p_value = 0.0
elif (a_better_b[1] > 0.05) and (b_better_a[1] < 0.05):
  best_model = "B"
  p_value = float(b_better_a[1])
else:
  raise ValueError("Statistical test failed")

# COMMAND ----------

data = [(best_model,((spark.sql("select current_timestamp()")).collect()[0][0]),p_value)]
data

# COMMAND ----------

df_pvalue  = spark.createDataFrame(data, T.StructType([
  T.StructField("best_model", T.StringType()),
  T.StructField("timestamp", T.TimestampType()),
  T.StructField("pvalue", T.FloatType())]
))

(
  df_pvalue
 .write
 .mode("append")
 .format("delta")
 .saveAsTable("solacc_ab_test.credit_risk_ab_testing")
)

# COMMAND ----------

(
  df_metrics
  .write
  .mode("append")
  .format("delta")
  .saveAsTable("solacc_ab_test.risk_metrics")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional: create a dashboard on Databricks SQL 
# MAGIC 
# MAGIC DB internal workspace link: https://e2-demo-west.cloud.databricks.com/sql/dashboards/02566bf1-3ecd-4d63-b3ba-b6ccf859a530-risk-demo
# MAGIC 
# MAGIC <img src="https://github.com/sergioballesterossolanas/databricks-ab-testing/blob/master/img/sql_dashboard.png?raw=true" width="1300"/>

# COMMAND ----------


