# Databricks notebook source
# MAGIC %md
# MAGIC # Live Stream
# MAGIC <img src="https://tdwi.org/articles/2017/08/07/-/media/TDWI/TDWI/BITW/datapipeline.jpg" width="700"/>
# MAGIC 
# MAGIC This notebook performs the following:
# MAGIC - Creates a streaming Delta table
# MAGIC - Injects rows into that table using the German Risk Data dataset
# MAGIC 
# MAGIC Leave this notebook running and we will read the stream from the following notebooks.
# MAGIC When we finish with the last notebook, we can cancel this run.

# COMMAND ----------

import time
import numpy as np

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS risk_stream_source;
# MAGIC CREATE TABLE risk_stream_source
# MAGIC   ( id INTEGER, 
# MAGIC     age INTEGER,
# MAGIC     sex STRING,
# MAGIC     job INTEGER,
# MAGIC     housing STRING,
# MAGIC     saving_accounts STRING,
# MAGIC     checking_account STRING,
# MAGIC     credit_amount INTEGER,
# MAGIC     duration INTEGER,
# MAGIC     purpose STRING
# MAGIC     )
# MAGIC   USING DELTA 

# COMMAND ----------

df = (
  spark
  .readStream
  .format("delta")
  .table("risk_stream_source")
)

display(df)

# COMMAND ----------

for next_row in range(600, 1000):
  time.sleep(np.random.uniform(0.1,0.4))
  print('Row inserted,', next_row)
  spark.sql(f"""
      INSERT INTO risk_stream_source (
      SELECT id, age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose FROM german_credit_data
      WHERE id = {next_row} )
  """)
  
